# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import os
import datetime
import time
import numpy as np
import yaml
import wandb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from loguru import logger

import utils
from dino_head import DINOHead
from vit3d import Vision_Transformer3D
from data_utils import make_dino_dataloaders

os.environ['WANDB_MODE'] = 'online'

# import lightning as L
vits_dict = {
    'vit_base': Vision_Transformer3D,
}

def make_dino_models(cfg):
    # ============ building student and teacher networks ... ============
    student = vits_dict[cfg['model']['arch']](
        **cfg['model']
    )
    cfg['model']['drop_path_rate'] = 0.0
    teacher = vits_dict[cfg['model']['arch']](
        **cfg['model'],
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

# def get_args():
#     # torchrun --nproc_per_node=3 dino_ddp.py --config_file configs_ddp.yaml --savename dino_experiment --seed 4844 --datasets IXI BRATS2023 OASIS3 --devices 1,2,3
#     parser = argparse.ArgumentParser(description='Train UDA model for MRI imaging for classification of AD')
#     parser.add_argument('--config_file', type=str, default='configs_ddp.yaml', help='Name of the config file')
#     parser.add_argument('--savename', type=str, help='Experiment name (used for saving files)')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
#     parser.add_argument('--datasets', type=str, nargs='+', default=['IXI'], help='List of datasets to use for training')
#     parser.add_argument('--devices', type=str, default="0,1,2,3", help='GPU devices to use')
#     parser.add_argument('--output_dir', type=str, default='./checkpoints/', help='Directory to save output files')
#     return parser.parse_args()

from dataclasses import dataclass, field
from typing import List
@dataclass
class ArgumentsDINO:
    config_file: str = field(default='configs_ddp_sweep.yaml', metadata={"help": "Name of the config file"})
    savename: str = field(default='dino_sweep', metadata={"help": "Experiment name (used for saving files)"})
    seed: int = field(default=4845, metadata={"help": "Random seed for reproducibility"})
    datasets: List[str] = field(default_factory=lambda: ['IXI'], metadata={"help": "List of datasets to use for training"})
    devices: str = field(default="0,1,2,3", metadata={"help": "GPU devices to use"})
    output_dir: str = field(default='./checkpoints/', metadata={"help": "Directory to save output files"})

def train_dino():
    with wandb.init(project="DAMIT_NEW[DINO]", tags="sweep", settings=wandb.Settings(init_timeout=120)) as run:
        sweeping_config = run.config
        print(f"Sweeping configuration: {sweeping_config}")
        # args = get_args()
        args = ArgumentsDINO()
        # Loads config file for fixed configs
        f_config = open(args.config_file,'rb')
        cfg = yaml.load(f_config, Loader=yaml.FullLoader)
        
        # assign sweeping config to cfg
        cfg['model']['out_dim'] = sweeping_config['out_dim']
        cfg['dino_head']['out_dim'] = sweeping_config['out_dim']
        cfg['dino']['norm_last_layer'] = sweeping_config['norm_last_layer']
        cfg['dino']['momentum_teacher'] = sweeping_config['momentum_teacher']
        cfg['dino']['teacher_temp'] = sweeping_config['teacher_temp']
        
        cfg['training']['epochs'] = sweeping_config['epochs']
        cfg['training']['batch_size_per_gpu'] = sweeping_config['batch_size_per_gpu']
        
        cfg['optimizer']['freeze_last_layer'] = sweeping_config['freeze_last_layer']
        cfg['optimizer']['lr'] = sweeping_config['lr']
        cfg['optimizer']['min_lr'] = sweeping_config['min_lr']
        cfg['training']['warmup_epochs'] = sweeping_config['warmup_epochs']
        cfg['optimizer']['optimizer'] = sweeping_config['optimizer']
        cfg['optimizer']['weight_decay'] = sweeping_config['weight_decay']
        
        cfg['transforms']['local_crop_img_size'] = sweeping_config['local_crop_img_size']
        cfg['transforms']['global_crops_scale'] = sweeping_config['global_crops_scale']
        cfg['transforms']['local_crops_scale'] = sweeping_config['local_crops_scale']
        cfg['transforms']['local_crops_number'] = sweeping_config['local_crops_number']
        cfg['transforms']['spacing'] = sweeping_config['spacing']
        
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
        # args.devices = [int(d) for d in args.devices.split(',')]
        
        os.makedirs(args.output_dir, exist_ok=True)
        # FILENAME = f"DINO_pt_{args.savename}_{'_'.join(args.datasets)}_seed_{args.seed}"
        # os.makedirs(os.path.join(args.output_dir, FILENAME), exist_ok=True)
        
        utils.init_distributed_mode(args)
        set_seed(args.seed)
        # print("git:\n  {}\n".format(utils.get_sha()))
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
        cudnn.benchmark = True
        
        # run = None
        # if utils.is_main_process():
        #     run = wandb.init(project="DAMIT_NEW[DINO]", name=FILENAME, config=cfg, dir=os.path.join(args.output_dir, FILENAME),
        #             tags=['DINO'], group=FILENAME)

        # logger.remove()
        # logger.add(os.path.join(args.output_dir, FILENAME, 'log.txt'))
        
        dataset, data_loader = make_dino_dataloaders(cfg, args.datasets)
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True) # new
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=cfg['training']['batch_size_per_gpu'],
            num_workers=cfg['data']['num_workers'],
            pin_memory=True,
            drop_last=True,
        )
        logger.success(f"Data loaded: there are {len(dataset)} images.")
        
        # ============ building student and teacher networks ... ============
        student_model, teacher_model, embed_dim = make_dino_models(cfg)
        # multi-crop wrapper handles forward with inputs of different resolutions
        student = utils.MultiCropWrapper(student_model, 
            DINOHead(norm_last_layer=True, **cfg['dino_head']
        ))
        teacher = utils.MultiCropWrapper(
            teacher_model,
            DINOHead(**cfg['dino_head']),
        )
        # move networks to gpu
        student, teacher = student.cuda(), teacher.cuda()
        # synchronize batch norms (if any)
        if utils.has_batchnorms(student):
            student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
            teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

            # we need DDP wrapper to have synchro batch norms working...
            teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[0])
            teacher_without_ddp = teacher.module
        else:
            # teacher_without_ddp and teacher are the same thing
            teacher_without_ddp = teacher
        student = nn.parallel.DistributedDataParallel(student, device_ids=[0])
        # teacher and student start with the same weights
        teacher_without_ddp.load_state_dict(student.module.state_dict())
        # there is no backpropagation through the teacher, so no need for gradients
        for p in teacher.parameters():
            p.requires_grad = False
        logger.success(f"Student and Teacher are built: they are both {cfg['model']['arch']} network.")

        # ============ preparing loss ... ============
        dino_loss = utils.DINOLoss(
            cfg['model']['out_dim'],
            cfg['transforms']['local_crops_number'] + 2,  # total number of crops = 2 global crops + local_crops_number
            cfg['dino']['warmup_teacher_temp'],
            cfg['dino']['teacher_temp'],
            cfg['dino']['warmup_teacher_temp_epochs'],
            cfg['training']['epochs'],
            ddp_mode=True,
            center_momentum=sweeping_config['center_momentum'],
        ).cuda()
        
        # ============ preparing optimizer ... ============
        params_groups = utils.get_params_groups(student)
        if cfg['optimizer']['optimizer'] == "adamw":
            optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        
        # for mixed precision training
        fp16_scaler = None
        if cfg['training']['use_fp16']:
            fp16_scaler = torch.cuda.amp.GradScaler()

        # ============ init schedulers ... ============
        lr = cfg['optimizer']['lr']
        min_lr = cfg['optimizer']['min_lr']
        weight_decay = cfg['optimizer']['weight_decay']
        weight_decay_end = cfg['optimizer']['weight_decay_end']
        batch_size_per_gpu = cfg['training']['batch_size_per_gpu']
        epochs = cfg['training']['epochs']
        warmup_epochs = cfg['training']['warmup_epochs']
        momentum_teacher = cfg['dino']['momentum_teacher']
        
        coeff_lr_div = float(cfg['training']['batch_size_per_gpu'] * utils.get_world_size()) # originally it is 256. == batch size
        lr_schedule = utils.cosine_scheduler(
            lr * (batch_size_per_gpu * utils.get_world_size()) / coeff_lr_div,  # linear scaling rule
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
        momentum_schedule = utils.cosine_scheduler(momentum_teacher, 1, epochs, len(data_loader))
        logger.success("Loss, optimizer and schedulers ready.")
        
        start_epoch = 0
        start_time = time.time()
        
        logger.warning("Starting DINO training !")
        for epoch in range(start_epoch, epochs):
            data_loader.sampler.set_epoch(epoch)

            # ============ training one epoch of DINO ... ============
            train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                epoch, run, fp16_scaler, cfg)

            # ============ writing logs ... ============
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': args,
                'cfg': cfg,
                'dino_loss': dino_loss.state_dict(),
            }
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            # utils.save_on_master(save_dict, os.path.join(args.output_dir, FILENAME, 'checkpoint.pth'))
            # if cfg['training']['saveckp_freq'] and epoch % cfg['training']['saveckp_freq'] == 0:
            #     utils.save_on_master(save_dict, os.path.join(args.output_dir, FILENAME, f'checkpoint{epoch:04}.pth'))
            if utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch}
                run.log(log_stats)
                # with (Path(f"{args.output_dir}/{FILENAME}") / "log.txt").open("a") as f:
                #     f.write(json.dumps(log_stats) + "\n")
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f'Training time {total_time_str}')
        # dist.destroy_process_group()
        # utils.save_on_master(save_dict, os.path.join(args.output_dir, FILENAME, 'checkpoint_last.pth'))
        
    ##############################################
    ##############################################  

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    wandb_run, fp16_scaler, cfg):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, cfg['training']['epochs'])
    for it, images in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

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
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        
        if wandb_run is not None:
            wandb_run.log({
                'iter_loss': loss.item(),
                'iter_lr': optimizer.param_groups[0]["lr"],
            }, step=it)
        
        torch.cuda.empty_cache()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    sweep_configurations = {
        "method": "bayes",
        "metric": {"name": "train_loss", "goal": "minimize"},
        "parameters": {
           "out_dim": {"values": [1024, 4096]},
           "norm_last_layer": {"values": [True, False]},
           "momentum_teacher": {"values": [0.996, 0.999, 0.9999]},
           "teacher_temp": {"values": [0.04, 0.07]},
           "center_momentum": {"values": [0.9, 0.95, 0.99]},
           "epochs": {"values": [40]},
           "batch_size_per_gpu": {"values": [8]},
           "freeze_last_layer": {"values": [1, 3]},
           "lr": {"values": [1e-5, 1e-3, 1e-2]},
           "min_lr": {"values": [1e-6, 1e-7]},
           "warmup_epochs": {"values": [10, 20]},
           "optimizer": {"values": ["adamw"]},
           "weight_decay": {"values": [0.4, 0.05, 0.001]},
           "local_crop_img_size": {"values": [[64, 64, 64], [80, 80, 80]]},
           "global_crops_scale": {"values": [[0.4, 1.], [0.7, 1.]]},
           "local_crops_scale": {"values": [[0.05, 0.4], [0.1, 0.6]]},
           "local_crops_number": {"values": [3, 6, 8]},
           "spacing": {"values": [[1.75, 1.75, 1.75], [1, 1, 1]]},
        }
    }
    # Process number
    logger.info(f"Process number: {os.getpid()}")
    # train_dino()
    # Initialize sweep
    sweep_id = "rag72uxi"
    if not sweep_id:
        sweep_id = wandb.sweep(sweep=sweep_configurations, project="DAMIT_NEW[DINO]")
    logger.info(f"Sweep ID: {sweep_id}")
    # Start the sweep agent
    wandb.agent(sweep_id, function=train_dino, project="DAMIT_NEW[DINO]")