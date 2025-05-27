# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 2023

@author: qasymjomart
"""

import numpy as np
import os
import argparse
import random
import yaml
from loguru import logger

import torch

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from lits import LitMAE
from dataloaders.make_dataloaders import make_mae_pretraining_dataloaders
from models.make_models import make_mae_model

# Set the seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print('Seed is set.')

def dir_exists():
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./wandb', exist_ok=True)
    os.makedirs('./monai_logs', exist_ok=True)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train UDA model for MRI imaging for classification of AD')
    parser.add_argument('--config_file', type=str, default='config_pretrain.yaml', help='Name of the config file')
    parser.add_argument('--model', default='mae', type=str, help='Pre-training model')
    parser.add_argument('--savename', type=str, help='Experiment name (used for saving files)')
    parser.add_argument('--datasets', nargs='+', type=str, help='Datasets to use for pre-training MAE')
    parser.add_argument('--seed', type=int, help='Experiment seed (for reproducible results)')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Mask ratio used for MAE')
    parser.add_argument('--use_aug', action='store_true')
    parser.add_argument('--devices', type=str, help='GPU devices to use')
    args = parser.parse_args()

    # Loads config file for fixed configs
    f_config = open(args.config_file,'rb')
    cfg = yaml.load(f_config, Loader=yaml.FullLoader)
    # print(torch.cuda.device_count())
    # Set seed
    set_seed(args.seed)
    dir_exists()

    # Set up GPU devices to use
    if cfg['TRAINING']['USE_GPU']:
        print(('Using GPU %s'%args.devices))
        os.environ["CUDA_DEVICE_ORDER"]=cfg['TRAINING']['CUDA_DEVICE_ORDER']
        os.environ["CUDA_VISIBLE_DEVICES"]=args.devices
    else:
        print('CPU mode')
    print('Process number: %d'%(os.getpid()))

    aug_prefix = 'aug' if args.use_aug else 'noaug'
    FILENAME_POSTFIX = f'{args.savename}_{args.model}_{int(args.mask_ratio*100)}_seed_{args.seed}'
    
    wandb_logger =  WandbLogger(project="[DAMIT NEW] Pre-training", 
                               name=FILENAME_POSTFIX, 
                               tags=[args.model, aug_prefix, str(args.mask_ratio), args.datasets], 
                               config=cfg)
    
    # Monai logs foldernames
    cfg['TRANSFORMS']['cache_dir_train'] = f'./monai_logs/pretrain_{FILENAME_POSTFIX}'
    
    if args.model == 'mae':
        model = make_mae_model(cfg, args)
        logger.success(f'MAE model {args.model} is created.')
        pretraining_dataloader, pretrain_dataset = make_mae_pretraining_dataloaders(cfg, args)
    
    # Save all configs and args (just in case)
    logger.info(cfg)
    logger.info(args)

    # create Lightning model
    model = LitMAE(model, 
                   mask_ratio=args.mask_ratio,
                   epochs=cfg['TRAINING']['EPOCHS'],
                   optimizer_hparams=cfg['SOLVER'])
    
    checkpoint_callback = ModelCheckpoint(
            monitor='loss',
            dirpath=f'checkpoints/{FILENAME_POSTFIX}/',
            filename='best-{epoch:02d}',
            save_top_k=1,
            mode='min',
            save_last=True,
            verbose=True
        )
    
    # trainer
    trainer = L.Trainer(
        accelerator='gpu',
        devices=[0],
        max_epochs=cfg['TRAINING']['EPOCHS'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    
    logger.success('Training is finished.')
