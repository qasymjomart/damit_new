# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import yaml
import wandb
import random
import utils
from loguru import logger
import torch
import torch.nn as nn
# import torch.distributed as dist
import torch.backends.cudnn as cudnn

from pathlib import Path
from ibot_head import iBOTHead
from evaluation import eval_pred

from vit3d import Vision_Transformer3D
from data_utils import make_ibot_dataloaders

vits_dict = {
    'vit_base': Vision_Transformer3D,
}

def make_ibot_models(cfg):
    # ============ building student and teacher networks ... ============
    student = vits_dict[cfg['model']['arch']](
        **cfg['model'],
        return_all_tokens=True,
        masked_im_modeling=cfg['ibot_model']['use_masked_im_modeling'],
    )
    cfg['model']['drop_path_rate'] = 0.0
    teacher = vits_dict[cfg['model']['arch']](
        **cfg['model'],
        return_all_tokens=True,
    )
    embed_dim = student.embed_dim
    return student, teacher, embed_dim

# Set the seed
def set_seed(seed):
    # L.utilities.seed.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print('Seed is set.')

def get_args():
    parser = argparse.ArgumentParser('iBOT', add_help=False)
    # torchrun --nproc_per_node=3 dino_ddp.py --config_file configs_ddp.yaml --savename dino_experiment --seed 4844 --datasets IXI BRATS2023 OASIS3 --devices 1,2,3
    parser = argparse.ArgumentParser(description='Train UDA model for MRI imaging for classification of AD')
    parser.add_argument('--config_file', type=str, default='configs_ddp.yaml', help='Name of the config file')
    parser.add_argument('--savename', type=str, help='Experiment name (used for saving files)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--datasets', type=str, nargs='+', default=['IXI'], help='List of datasets to use for training')
    parser.add_argument('--devices', type=str, default="0,1,2,3", help='GPU devices to use')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/', help='Directory to save output files')
    return parser

def train_ibot():
    args = get_args()
    # Loads config file for fixed configs
    f_config = open(args.config_file,'rb')
    cfg = yaml.load(f_config, Loader=yaml.FullLoader)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    args.devices = [int(d)-1 for d in args.devices.split(',')]
    
    os.makedirs(args.output_dir, exist_ok=True)
    FILENAME = f"iBOT_pt_{args.savename}_{'_'.join(args.datasets)}_seed_{args.seed}"
    run = wandb.init(project="DAMIT_NEW[DINO]", name=FILENAME, config=cfg, dir=os.path.join(args.output_dir, FILENAME),
               tags=['iBOT'], group=FILENAME)
    os.makedirs(os.path.join(args.output_dir, FILENAME), exist_ok=True)
    
    utils.init_distributed_mode(args)
    set_seed(args.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
        
    # ============ preparing data ... ============
    dataset, sampler, data_loader = make_ibot_dataloaders(cfg, args.datasets)    
    print(f"Data loaded: there are {len(dataset)} images.")

    ### LAST STOPPED HERE AS OF July 14, 6:53pm ###

    # ============ building student and teacher networks ... ============
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    student_model, teacher_model, embed_dim = make_ibot_models(cfg)

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student_model, iBOTHead(
        in_dim=embed_dim,
        out_dim=cfg['ibot_head']['out_dim'],
        patch_out_dim=cfg['ibot_head']['patch_out_dim'],
        norm=cfg['ibot_head']['norm_in_head'],
        act=cfg['ibot_head']['act_in_head'],
        norm_last_layer=cfg['ibot_head']['norm_last_layer'],
        shared_head=cfg['ibot_head']['shared_head'],
    ))
    teacher = utils.MultiCropWrapper(
        teacher_model,
        iBOTHead(
            in_dim=embed_dim, 
            out_dim=cfg['ibot_head']['out_dim'],
            patch_out_dim=cfg['ibot_head']['patch_out_dim'],
            norm=cfg['ibot_head']['norm_in_head'],
            act=cfg['ibot_head']['act_in_head'],
            shared_head=cfg['ibot_head']['shared_head'],
        ),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    logger.success(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    same_dim = cfg['ibot_head']['shared_head'] or cfg['ibot_head']['shared_head_teacher']
    ibot_loss = utils.iBOTLoss(
        cfg['ibot_head']['out_dim'],
        cfg['ibot_head']['out_dim'] if same_dim else cfg['ibot_head']['patch_out_dim'],
        cfg['transforms']['global_crops_number'],
        cfg['transforms']['local_crops_number'],
        cfg['ibot']['warmup_teacher_temp'],
        cfg['ibot']['teacher_temp'],
        cfg['ibot']['warmup_teacher_patch_temp'],
        cfg['ibot']['teacher_patch_temp'],
        cfg['ibot']['warmup_teacher_temp_epochs'],
        cfg['training']['epochs'],
        lambda1=cfg['ibot_model']['lambda1'],
        lambda2=cfg['ibot_model']['lambda2'],
        mim_start_epoch=cfg['ibot_model']['pred_start_epoch'],
    ).cuda() # should be already configured for DDP training
        
    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr = cfg['optimizer']['lr']
    min_lr = cfg['optimizer']['min_lr']
    weight_decay = cfg['optimizer']['weight_decay']
    weight_decay_end = cfg['optimizer']['weight_decay_end']
    batch_size_per_gpu = cfg['training']['batch_size_per_gpu']
    epochs = cfg['training']['epochs']
    warmup_epochs = cfg['training']['warmup_epochs']
    momentum_teacher = cfg['ibot']['momentum_teacher']
    
    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        lr * (batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        min_lr,
        epochs, len(data_loader),
        warmup_epochs=warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        weight_decay,
        weight_decay_end,
        epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(momentum_teacher, 1,
                                            epochs, len(data_loader))            
    logger.success("Loss, optimizer and schedulers ready.")
    
    start_epoch = 0
    start_time = time.time()

    print("Starting iBOT training!")
    for epoch in range(start_epoch, cfg['training']['epochs']):
        data_loader.sampler.set_epoch(epoch)
        data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch of iBOT ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, cfg)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'cfg': cfg,
            'ibot_loss': ibot_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, FILENAME, 'checkpoint.pth'))
        if cfg['training']['saveckp_freq'] and epoch % cfg['training']['saveckp_freq'] == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, FILENAME, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        run.log(log_stats, step=epoch)
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    ###########################
    ###########################

def train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, cfg):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, cfg['training']['epochs'])
    
    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

    pred_labels, real_labels = [], []
    for it, (images, labels, masks) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]        
        
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # get global views
            teacher_output = teacher(images[:cfg['transforms']['global_crops_number']])
            student_output = student(images[:cfg['transforms']['global_crops_number']], \
                mask=masks[:cfg['transforms']['global_crops_number']])
            
            # get local views
            student.module.backbone.masked_im_modeling = False
            student_local_cls = student(images[cfg['transforms']['global_crops_number']:])[0] if len(images) > cfg['transforms']['global_crops_number'] else None
            student.module.backbone.masked_im_modeling = cfg['ibot_model']['use_masked_im_modeling']

            all_loss = ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch)
            loss = all_loss.pop('loss')

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # log statistics
        probs1 = teacher_output[0].chunk(cfg['transforms']['global_crops_number'])
        probs2 = student_output[0].chunk(cfg['transforms']['global_crops_number'])
        pred1 = utils.concat_all_gather(probs1[0].max(dim=1)[1]) 
        pred2 = utils.concat_all_gather(probs2[1].max(dim=1)[1])
        acc = (pred1 == pred2).sum() / pred1.size(0)
        pred_labels.append(pred1)
        real_labels.append(utils.concat_all_gather(labels.to(pred1.device)))

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if cfg['optimizer']['clip_grad']:
                param_norms = utils.clip_gradients(student, cfg['optimizer']['clip_grad'])
            utils.cancel_gradients_last_layer(epoch, student,
                                              cfg['optimizer']['freeze_last_layer'])
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if cfg['optimizer']['clip_grad']:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, cfg['optimizer']['clip_grad'])
            utils.cancel_gradients_last_layer(epoch, student,
                                              cfg['optimizer']['freeze_last_layer'])
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(acc=acc)

    pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    real_labels = torch.cat(real_labels).cpu().detach().numpy()
    nmi, ari, fscore, adjacc = eval_pred(real_labels, pred_labels, calc_acc=False)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("NMI: {}, ARI: {}, F: {}, ACC: {}".format(nmi, ari, fscore, adjacc))
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_dict.update({"nmi": nmi, "ari": ari, "fscore": fscore, "adjacc": adjacc})
    return return_dict

if __name__ == '__main__':
    # Process number
    logger.info(f"Process number: {os.getpid()}")
    train_ibot()