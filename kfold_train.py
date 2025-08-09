# -*- coding: utf-8 -*-
"""
Created on Sun May 28 2023

@author: qasymjomart
"""
import os
import numpy as np
import pandas as pd
import shutil
from datetime import datetime
import argparse
from loguru import logger
import random
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import time
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from lits import LitViT
from dataloaders.make_dataloaders import make_kfold_dataloaders, replace_data_path
from models.make_models import make_model
# from utils.utils import load_pretrained_checkpoint
from utils.prepare_model import prepare_model_for_training

import warnings 
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

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

if __name__ == '__main__':

    # Parse some variable configs
    parser = argparse.ArgumentParser(description='Train UDA model for MRI imaging for classification of AD')
    parser.add_argument('--config_file', type=str, default='configs/config_vitb.yaml', help='Name of the config file')
    parser.add_argument('--savename', type=str, help='Experiment name (used for saving files)')
    parser.add_argument('--classes_to_use', nargs='+', type=str, help='Classes to use (enter by separating by space, e.g. CN AD MCI)')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--seed', type=int, help='Experiment seed (for reproducible results)')
    parser.add_argument('--devices', type=str, help='GPU devices to use')
    parser.add_argument('--patch_embed_fun', type=str, default='conv3d', help='Patch embed function to use')
    parser.add_argument('--checkpoint', default='./checkpoints/', type=str, help='Checkpoint model path')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate to use')
    parser.add_argument('--drop_path', type=float, default=0.1, help='Drop_path to use')
    parser.add_argument('--attn_p', type=float, default=0.1, help='Attn_p dropout to use')
    parser.add_argument('--p', type=float, default=0.0, help='Dropout rate to use')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size for ViT transformer')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Learning rate scheduler')
    parser.add_argument('--train_size', type=str, default='all', help='Train size: [0.2, 0.4, 0.6, 0.8, all]')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--vit_size', type=str, default='base', help='ViT base, small, large')
    parser.add_argument('--use_aug', action='store_true')
    parser.add_argument('--use_pretrained', type=str, help='Path to pre-trained model checkpoint to load')
    args = parser.parse_args()

    # Loads config file for fixed configs
    f_config = open(args.config_file,'rb')
    cfg = yaml.load(f_config, Loader=yaml.FullLoader) 

    # Set seed
    set_seed(args.seed)

    # Set up GPU devices to use
    if cfg['TRAINING']['USE_GPU']:
        print(f'Using GPU {args.devices}')
        os.environ["CUDA_DEVICE_ORDER"]=cfg['TRAINING']['CUDA_DEVICE_ORDER']
        os.environ["CUDA_VISIBLE_DEVICES"]=args.devices
    else:
        print('CPU mode')
    print(f'Process number: {os.getpid()} \n')

    df = pd.read_csv(replace_data_path(cfg[args.dataset]['labelsroot']))
    df = df[df['Group'].isin(args.classes_to_use)]

    cfg['MODEL']['patch_embed_fun'] = args.patch_embed_fun
    cfg['MODEL']['patch_size'] = args.patch_size
    cfg['TRAINING']['EPOCHS'] = args.epochs
    cfg['SOLVER']['optimizer'] = args.optimizer
    cfg['SOLVER']['lr'] = args.lr
    cfg['SOLVER']['scheduler'] = args.scheduler
    cfg['DATALOADER']['train_size'] = args.train_size if args.train_size == 'all' else float(args.train_size)
    cfg['DATALOADER']['BATCH_SIZE'] = args.batch_size
    cfg['MODEL']['drop_path_rate'] =args.drop_path
    cfg['MODEL']['attn_p'] = args.attn_p
    cfg['MODEL']['p'] = args.p
    cfg['MODEL']['n_classes'] = len(args.classes_to_use)

    kfold_results = {"val_accs": [],
                     "best_epoch": [],
                     "recalls": [],
                     "f1s": [],
                     "corrects": [],
                     "n_datapoints": [],
                     "ratios": []}

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.seed)
    
    FILENAME = f'{args.savename}_{args.dataset}_seed_{args.seed}'
    
    # Init wandb
    wandb_logger = WandbLogger(project="DAMIT_NEW", 
                               name=FILENAME, 
                               tags=f'{args.savename}_{args.dataset}', config=cfg)

    for i, (train_index, test_index) in enumerate(skf.split(df, df['Group'])):
        # FILENAME_POSTFIX = args.savename + '_' + args.mode + '_seed_' + str(args.seed)
        timestamp_current = datetime.now()
        timestamp_current = timestamp_current.strftime("%Y%m%d_%H%M")
        
        FILENAME_POSTFIX = f'{FILENAME}_fold_{i}'
        logger.add(f'./logs/{FILENAME_POSTFIX}_{timestamp_current}.log', rotation="10 MB", level='TRACE')

        # Monai logs foldernames
        cfg['TRANSFORMS']['cache_dir_train'] = f'./monai_logs/train_{FILENAME_POSTFIX}'
        cfg['TRANSFORMS']['cache_dir_test'] = f'./monai_logs/test_{FILENAME_POSTFIX}'
        # Number of classes to use
        
        # Set up logger file
        logger.info(f'Process number: {os.getpid()}')
        logger.info(f"Started training. Savename : {args.savename}")
        logger.info(f"Seed : {args.seed}")
        logger.info(f"dataset dataset : {args.dataset}")

        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]

        logger.info(f'Fold {i}: ')
        logger.info(f'Train: {len(df_train)}, Test: {len(df_test)}')
        logger.info(f'Train balance: {df_train["Group"].value_counts()}')
        logger.info(f'Test balance: {df_test["Group"].value_counts()}')

        train_dataloader, test_dataloader, train_dataset, test_dataset, ratios_train, ratios_test = make_kfold_dataloaders(cfg, args, df_train, df_test)

        del df_train, df_test
    
        logger.info(f'Number of classes to be used: {args.classes_to_use}, {cfg["MODEL"]["n_classes"]}')
        logger.info(f'Train labels ratio: {ratios_train}, Test labels ratio: {ratios_test}')
        logger.info(f'Train ratios (%): {[round(100*x/sum(ratios_train.values()), 2) for x in ratios_train.values()]}, Test ratios (%): {[round(100*x/sum(ratios_test.values()), 2) for x in ratios_test.values()]}')
        logger.info(f'Train set labels ratio: {ratios_train}')
        logger.info(f'Test set labels ratio: {ratios_test}')

        ### MODEL ####
        model = make_model(cfg, args)
        
        # Load pre-trained model weights (MAE)
        # if args.use_pretrained is not None:
        #     model = load_pretrained_checkpoint(model, args.use_pretrained, cfg['TRAINING']['CHECKPOINT_TYPE'])
    
        # Move model to GPU
        # if cfg['TRAINING']['USE_GPU']:
        #     model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        #     params = sum([np.prod(p.size()) for p in model_parameters])
        #     logger.info(f'Num of parameters in the model: {params}')
        #     model.cuda()
        
        model = prepare_model_for_training(model, cfg)
        
        # Initialize loss function (with weight balance)
        class_numbers = []
        for class_name in args.classes_to_use:
            class_numbers += [args.classes_to_use.index(class_name)] * ratios_train[class_name]
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(class_numbers), y=class_numbers)
        class_weights = torch.Tensor(class_weights).cuda()
        logger.info(f'Class weights: {class_weights}')
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Save all configs and args just in case
        logger.info(cfg)
        logger.info(args)

        lit_vit = LitViT(model, cfg['MODEL']['n_classes'],
                         pretrained_model_path=args.use_pretrained,
                         loss_fn=criterion, learning_rate=cfg['SOLVER']['lr'],
                         weight_decay=cfg['SOLVER']['weight_decay'], 
                         betas=(cfg['SOLVER']['beta1'], cfg['SOLVER']['beta2']),
                         epochs=args.epochs,
                         mode=cfg['MODE'])
        
        if os.path.exists(f'checkpoints/{FILENAME}/fold_{i}'):
            logger.info(f'Checkpoint folder for fold {i} already exists. Removing it.')
            # pause for 5 sec
            time.sleep(5)
            shutil.rmtree(f'checkpoints/{FILENAME}/fold_{i}')
        
        os.makedirs(f'checkpoints/{FILENAME}/fold_{i}')
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            dirpath=f'checkpoints/{FILENAME}/fold_{i}/',
            filename='best-{epoch:02d}',
            save_weights_only=True,
            save_top_k=1,
            mode='max',
            save_last=True,
            verbose=True
        )
                
        trainer = L.Trainer(max_epochs=cfg['TRAINING']['EPOCHS'],
                            # default_root_dir=f'checkpoints/{FILENAME}/fold_{i}/',
                            accelerator='gpu',
                            devices=[0],
                            # precision='16-mixed',
                            num_sanity_val_steps=0,
                            # precision=16,
                            logger=wandb_logger,
                            # profiler='simple',
                            callbacks=[checkpoint_callback],
                            )
        
        trainer.fit(lit_vit, train_dataloader, val_dataloaders=test_dataloader)
        
        kfold_results["val_accs"].append(max(lit_vit.val_accs))
        # recall, f1, and corrects will be selected from the best epoch which is max val_acc
        best_epoch = lit_vit.val_accs.index(max(lit_vit.val_accs))
        kfold_results["best_epoch"].append(best_epoch)
        kfold_results["recalls"].append(lit_vit.recalls[best_epoch])
        kfold_results["f1s"].append(lit_vit.f1s[best_epoch])
        kfold_results["corrects"].append(lit_vit.corrects[best_epoch])
        n_datapoints = len(test_dataset)
        kfold_results["n_datapoints"].append(n_datapoints)
        kfold_results["ratios"].append(ratios_test)
        
        logger.info(kfold_results)

        del model, train_dataloader, test_dataloader, train_dataset, test_dataset

        shutil.rmtree(f'./monai_logs/train_{FILENAME_POSTFIX}')
        shutil.rmtree(f'./monai_logs/test_{FILENAME_POSTFIX}')

print(kfold_results)
print(f'k-fold acc: {sum(kfold_results["corrects"])/sum(kfold_results["n_datapoints"]):.2f}')
print(f'avg of val accs {round(sum(kfold_results["val_accs"])/4, 2)} ± {round(np.std(kfold_results["val_accs"]), 2)}')
print(f'avg of recalls {round(sum(kfold_results["recalls"])/4, 2)} ± {round(np.std(kfold_results["recalls"]), 2)}')
print(f'avg of f1s {round(sum(kfold_results["f1s"])/4, 2)} ± {round(np.std(kfold_results["f1s"]), 2)}')

# save kfold results acc into a file
with open(f'./results/kfold_results_{FILENAME}.txt', 'w') as f:
    f.write(f'{FILENAME}\n')
    f.write(str(kfold_results) + '\n')
    f.write(f'k-fold acc: {sum(kfold_results["corrects"])/sum(kfold_results["n_datapoints"]):.2f}')
    f.write(f'avg of val accs {round(sum(kfold_results["val_accs"])/4, 2)} ± {round(np.std(kfold_results["val_accs"]), 2)}')
    f.write(f'avg of recalls {round(sum(kfold_results["recalls"])/4, 2)} ± {round(np.std(kfold_results["recalls"]), 2)}')
    f.write(f'avg of f1s {round(sum(kfold_results["f1s"])/4, 2)} ± {round(np.std(kfold_results["f1s"]), 2)}')
    f.write('\n')
    f.write('\n')