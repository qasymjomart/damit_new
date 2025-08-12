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
import argparse
import random
import os
import sys
import numpy as np
import yaml
import math
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger

import utils
from dino_head import DINOHead
from vit3d import Vision_Transformer3D
from data_utils import make_dino_dataloaders

import lightning as L
vits_dict = {
    'vit_base': Vision_Transformer3D,
}

# Set the seed
def set_seed(seed):
    L.utilities.seed.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print('Seed is set.')
    
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

class LitDINO(L.LightningModule):
    def __init__(self, student_model, 
                 teacher_model,
                 data_loader,
                 cfg):
        super().__init__()
        # dino parameters
        out_dim = cfg['model']['out_dim']
        local_crops_number = cfg['transforms']['local_crops_number']
        momentum_teacher = cfg['dino']['momentum_teacher']
        teacher_temp = cfg['dino']['teacher_temp']
        
        # optimizer parameters
        lr = cfg['optimizer']['lr']
        min_lr = cfg['optimizer']['min_lr']
        weight_decay = cfg['optimizer']['weight_decay']
        weight_decay_end = cfg['optimizer']['weight_decay_end']
        self.freeze_last_layer = cfg['optimizer']['freeze_last_layer']
        self.clip_grad = cfg['optimizer']['clip_grad']
        
        # training parameters
        batch_size = cfg['training']['batch_size']
        epochs = cfg['training']['epochs']
        warmup_epochs = cfg['training']['warmup_epochs']
        warmup_teacher_temp = cfg['dino']['warmup_teacher_temp']
        warmup_teacher_temp_epochs = cfg['dino']['warmup_teacher_temp_epochs']
        
        # multi-crop wrapper handles forward with inputs of different resolutions
        self.student = utils.MultiCropWrapper(student_model, 
            DINOHead(norm_last_layer=True, **cfg['dino_head']
        ))
        self.teacher = utils.MultiCropWrapper(
            teacher_model,
            DINOHead(**cfg['dino_head']),
        )
        # teacher and student start with the same weights
        self.teacher.load_state_dict(self.student.state_dict())
        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        print("Student and Teacher are built.")
        
        # ============ preparing loss ... ============
        self.dino_loss = utils.DINOLoss(
            out_dim,
            local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            warmup_teacher_temp,
            teacher_temp,
            warmup_teacher_temp_epochs,
            epochs,
        )
        
        # ============ init schedulers ... ============
        self.lr_schedule = utils.cosine_scheduler(
            lr * (batch_size) / 256.,  # linear scaling rule
            min_lr, epochs, len(data_loader), warmup_epochs=warmup_epochs)
        
        self.wd_schedule = utils.cosine_scheduler(weight_decay, weight_decay_end, epochs, len(data_loader))
        # momentum parameter is increased to 1. during training with a cosine schedule
        self.momentum_schedule = utils.cosine_scheduler(momentum_teacher, 1,
                                                epochs, len(data_loader))
        print("Loss, optimizer and schedulers ready.")
    
    def on_train_batch_start(self, batch, batch_idx):
        it = self.global_step
        for i, param_group in enumerate(self.optimizers().param_groups):
            param_group["lr"] = self.lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[it]       
    
    def training_step(self, batch, batch_idx):
        images = batch
        ### teacher and student forward passes + compute dino loss ###
        teacher_output = self.teacher(images[:2])  # only the 2 global views pass through the teacher
        student_output = self.student(images)
        loss = self.dino_loss(student_output, teacher_output, self.current_epoch)
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)
        
        self.log_dict({'loss': loss,
                       'lr': self.optimizers().param_groups[0]["lr"],
                       'wd': self.optimizers().param_groups[0]["weight_decay"]},
                      prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        params_groups = utils.get_params_groups(self.student)
        optimizer = torch.optim.AdamW(params_groups)
        return optimizer
    
    def on_before_optimizer_step(self, optimizer):
        ### If using AMP, the gradients are already scaled before this hook
        # If clipping gradients, they are not clipped yet before this hook
        ### param_norms = None
        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=self.clip_grad, gradient_clip_algorithm="norm")
        # utils.cancel_gradients_last_layer(self.current_epoch, self.student, self.freeze_last_layer)
        if self.current_epoch < self.freeze_last_layer:
            for n, p in self.student.named_parameters():
                if "last_layer" in n:
                    p.grad = None
    
    def on_train_epoch_end(self):
        # EMA update for the teacher
        with torch.no_grad():
            m = self.momentum_schedule[self.global_step-1]
            for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        torch.cuda.synchronize()

def get_args():
    parser = argparse.ArgumentParser(description='Train UDA model for MRI imaging for classification of AD')
    parser.add_argument('--config_file', type=str, default='configs.yaml', help='Name of the config file')
    parser.add_argument('--savename', type=str, help='Experiment name (used for saving files)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--datasets', type=str, nargs='+', default=['IXI'], help='List of datasets to use for training')
    parser.add_argument('--devices', type=str, default="0,1,2,3", help='GPU devices to use')
    return parser.parse_args()

def train_dino():
    args = get_args()
    # Loads config file for fixed configs
    f_config = open(args.config_file,'rb')
    cfg = yaml.load(f_config, Loader=yaml.FullLoader)
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.devices
    
    dataset, data_loader = make_dino_dataloaders(cfg, args.datasets)
    student_model, teacher_model, embed_dim = make_dino_models(cfg)
    
    model = LitDINO(student_model, teacher_model,
                    data_loader, cfg)
    
    FILENAME = f"DINO_pt_{args.savename}_{'_'.join(args.datasets)}_seed_{args.seed}"
    
    # Init wandb
    wandb_logger = WandbLogger(project="DAMIT_NEW[DINO]", 
                               name=FILENAME, 
                               tags=f'{args.savename}_{args.datasets}', config=cfg)
    
    os.makedirs(f'checkpoints/{FILENAME}', exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
            monitor='loss',
            dirpath=f'checkpoints/{FILENAME}/',
            filename='best-{epoch:02d}',
            save_weights_only=True,
            save_top_k=1,
            mode='min',
            save_last=True,
            verbose=True
        )
    
    trainer = L.Trainer(
        accelerator='gpu',
        devices=[0],
        logger=wandb_logger,
        max_epochs=cfg['training']['epochs'],
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],
    )
    
    trainer.fit(model, data_loader)
    
    logger.success(f"Training completed. Model saved to checkpoints/{FILENAME}/best.ckpt")

if __name__ == "__main__":
    # Process number
    logger.info(f"Process number: {os.getpid()}")
    train_dino()