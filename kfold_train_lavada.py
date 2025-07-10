# -*- coding: utf-8 -*-
"""
Created on Sun May 28 2023

@author: qasymjomart
"""

from __future__ import print_function, division
import numpy as np
import pandas as pd
import shutil

import os
import gc
import json
from datetime import datetime
import argparse
from loguru import logger as loguru_logger
import random
import yaml
import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.cuda import amp

from dataloaders.make_dataloaders import make_kfold_dataloaders
from models.make_models import make_model
from do_train import do_train, do_inference
from utils.utils import load_pretrained_checkpoint
from utils.optimizers import make_optimizer
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

def wandb_setup(cfg, args, FILENAME_POSTFIX):
    # start a new wandb run to track this script
    os.makedirs('./wandb/'+FILENAME_POSTFIX+'/', exist_ok=True)
    wandb.login(key='9216b9bab31599e85cbbd6a62dda77c7e61f4552')
    wandb.init(
        # set the wandb project where this run will be logged
        project="[DAMIT OTHERS] K-Fold Training",
        name=FILENAME_POSTFIX,
        
        # track hyperparameters and run metadata
        config={
        "config_file": args.config_file,
        "dataset": args.dataset,
        "pre-trained model": None,
        "lr": cfg['SOLVER']['lr'],
        "checkpoint_type": cfg['TRAINING']['CHECKPOINT_TYPE'],
        "dir": "./wandb/"+FILENAME_POSTFIX+"/"
        }
    )

if __name__ == '__main__':

    # Parse some variable configs
    parser = argparse.ArgumentParser(description='Train UDA model for MRI imaging for classification of AD')
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Name of the config file')
    parser.add_argument('--savename', type=str, help='Experiment name (used for saving files)')
    parser.add_argument('--classes_to_use', nargs='+', type=str, help='Classes to use (enter by separating by space, e.g. CN AD MCI)')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--seed', type=int, help='Experiment seed (for reproducible results)')
    parser.add_argument('--devices', type=str, help='GPU devices to use')
    parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count of training')
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

    df = pd.read_csv(cfg[args.dataset]['labelsroot'])
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

    kfold_results = {"acc": [],
                     "bal_acc": [],
                     "auc": [],
                     "corrects": [],
                     "n_datapoints": [],
                     "ratios": []}

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.seed)
    
    # Init wandb
    wandb_setup(cfg, args, f'{args.savename}_{args.dataset}_seed_{args.seed}')

    # load LAVADA data for correct stratified k-fold division
    lavada_data = json.load(open(f"/home/guest/qasymjomart/lada/data_generation/data/binary_{args.dataset.lower()}_llama33_ftdata_1may2025.json"))
    print(f'Len of total data: {len(lavada_data)}')
    lavada_data_df = pd.DataFrame(lavada_data)
    
    
    for i, (train_index, test_index) in enumerate(skf.split(lavada_data_df, lavada_data_df['label'])):
        # FILENAME_POSTFIX = args.savename + '_' + args.mode + '_seed_' + str(args.seed)
        timestamp_current = datetime.now()
        timestamp_current = timestamp_current.strftime("%Y%m%d_%H%M")
        
        FILENAME_POSTFIX = f'{args.savename}_{args.dataset}_seed_{args.seed}_fold_{i}'
        loguru_logger.add(f'./logs/{FILENAME_POSTFIX}_{timestamp_current}.log', rotation="10 MB", level='TRACE')

        # Monai logs foldernames
        cfg['TRANSFORMS']['cache_dir_train'] = f'./monai_logs/train_{FILENAME_POSTFIX}'
        cfg['TRANSFORMS']['cache_dir_test'] = f'./monai_logs/test_{FILENAME_POSTFIX}'
        # Number of classes to use
        
        # Set up logger file
        loguru_logger.info(f'Process number: {os.getpid()}')
        loguru_logger.info(f"Started training. Savename : {args.savename}")
        loguru_logger.info(f"Seed : {args.seed}")
        loguru_logger.info(f"dataset dataset : {args.dataset}")

        # subject ids from total_data_df
        train_subject_ids = lavada_data_df.iloc[train_index]["custom_id"].values
        test_subject_ids = lavada_data_df.iloc[test_index]["custom_id"].values
        
        train_subject_ids = [sample[5:] for sample in train_subject_ids]
        test_subject_ids = [sample[5:] for sample in test_subject_ids]
        
        df_train = df[df["Subject"].isin(train_subject_ids)]
        df_test = df[df["Subject"].isin(test_subject_ids)]
    
        # df_train = df.iloc[train_index]
        # df_test = df.iloc[test_index]

        loguru_logger.info(f'Fold {i}: ')
        loguru_logger.info(f'Train: {len(df_train)}, Test: {len(df_test)}')
        loguru_logger.info(f'Train balance: {df_train["Group"].value_counts()}')
        loguru_logger.info(f'Test balance: {df_test["Group"].value_counts()}')

        train_dataloader, test_dataloader, train_dataset, test_dataset, ratios_train, ratios_test = make_kfold_dataloaders(cfg, args, df_train, df_test)

        del df_train, df_test
        gc.collect()
    
        loguru_logger.info(f'Number of classes to be used: {args.classes_to_use}, {cfg["MODEL"]["n_classes"]}')
        loguru_logger.info(f'Train labels ratio: {ratios_train}, Test labels ratio: {ratios_test}')
        loguru_logger.info(f'Train ratios (%): {[round(100*x/sum(ratios_train.values()), 2) for x in ratios_train.values()]}, Test ratios (%): {[round(100*x/sum(ratios_test.values()), 2) for x in ratios_test.values()]}')
        loguru_logger.info(f'Train set labels ratio: {ratios_train}')
        loguru_logger.info(f'Test set labels ratio: {ratios_test}')

        ### MODEL ####
        model = make_model(cfg, args)
        
        # Load pre-trained model weights (MAE)
        if args.use_pretrained is not None:
            model = load_pretrained_checkpoint(model, args.use_pretrained, cfg['TRAINING']['CHECKPOINT_TYPE'])
    
        # Move model to GPU
        if cfg['TRAINING']['USE_GPU']:
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            loguru_logger.info(f'Num of parameters in the model: {params}')
            model.cuda()
        
        model = prepare_model_for_training(model, cfg)

        optimizer = make_optimizer(cfg, args, model)
        scaler = amp.GradScaler()
        
        # Initialize loss function (with weight balance) and optimizer
        # weight_balance = torch.Tensor(list(ratios_train.values())).cuda()
        # criterion = nn.CrossEntropyLoss(weight=weight_balance)
        class_numbers = ratios_train['CN']*[0] + ratios_train['AD']*[1]
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(class_numbers), y=class_numbers)
        class_weights = torch.Tensor(class_weights).cuda()
        loguru_logger.info(f'Class weights: {class_weights}')
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # if len([x for x in args.devices.split(",")]) > 1: # if more than 1 GPU selected
        #     model = torch.nn.DataParallel(model)
        #     loguru_logger.info('Multi-GPU training enabled.')
        
        # Save all configs and args just in case
        loguru_logger.info(cfg)
        loguru_logger.info(args)


        trained_model = do_train(
            cfg, args, FILENAME_POSTFIX, model, criterion, optimizer, scaler, train_dataloader,
            train_dataset, loguru_logger, True, 
            test_dataloader
        )
        
        test_acc, bal_acc, test_auc, corrects, n_datapoints = do_inference(
            cfg, args, trained_model, test_dataloader, loguru_logger, False
        )

        kfold_results["acc"].append(test_acc)
        kfold_results["bal_acc"].append(round(bal_acc, 2))
        kfold_results["auc"].append(round(test_auc, 2))
        kfold_results["corrects"].append(corrects)
        kfold_results["n_datapoints"].append(n_datapoints)
        kfold_results["ratios"].append(ratios_test)

        loguru_logger.info(kfold_results)

        del model, trained_model, train_dataloader, test_dataloader, train_dataset, test_dataset

        shutil.rmtree(f'./monai_logs/train_{FILENAME_POSTFIX}')
        shutil.rmtree(f'./monai_logs/test_{FILENAME_POSTFIX}')

        if i == 3:
            try:
                wandb.log({"k_fold acc": round(100*sum(kfold_results["corrects"])/sum(kfold_results["n_datapoints"]), 2)})
            except:
                pass

    wandb.finish()

print(kfold_results)
print(f'k-fold acc: {round(100*sum(kfold_results["corrects"])/sum(kfold_results["n_datapoints"]), 2)}')
print(f'avg of test accs {round(sum(kfold_results["acc"])/4, 2)}')
# print(f'avg of test calibration errors {round(sum(kfold_results["calib"])/4, 2)}')
print(f'avg bal acc {round(sum(kfold_results["bal_acc"])/4, 2)}')
print(f'avg auc {round(sum(kfold_results["auc"])/4, 2)}')

# save kfold results acc into a file (append it meaning that if the file exists, it will add the results to the end of the file)
with open(f'./results/kfold_results_bank_{args.savename}.txt', 'a') as f:
    f.write(f'{args.savename}_{args.dataset}_{args.seed}\n')
    f.write(str(kfold_results) + '\n')
    f.write(f'k-fold acc: {round(100*sum(kfold_results["corrects"])/sum(kfold_results["n_datapoints"]), 2)}\n')
    f.write(f'avg of test accs {round(sum(kfold_results["acc"])/4, 2)}\n')
    f.write(f'avg bal acc {round(sum(kfold_results["bal_acc"])/4, 2)}\n')
    f.write(f'avg auc {round(sum(kfold_results["auc"])/4, 2)}\n')
    f.write('\n')
    f.write('\n')