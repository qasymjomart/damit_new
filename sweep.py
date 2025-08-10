# -*- coding: utf-8 -*-
"""
Created on Sun Aug 9 2025
@author: qasymjomart
"""
import os
import numpy as np
import pandas as pd
import random
import yaml
import shutil
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import wandb

import torch
import torch.nn as nn
import lightning as L

from dataclasses import dataclass, field
from typing import List

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

@dataclass
class ArgumentsMimic:
    use_aug: bool = True
    classes_to_use: List[str] = field(default_factory=lambda: ['CN', 'AD'])
    dataset: str = 'ADNI2'
    seed: int = 5426
    vit_size: str = 'base'
    use_pretrained: str = None
    

def train_kfold(sweeping_config):
    RUN_NAME = "-".join(str(f"{k}_{v}") for k, v in sweeping_config.items())
    print(f"Running sweep with config: {RUN_NAME}")
    print(f"Sweeping config: {sweeping_config}")
    print()
    
    args = ArgumentsMimic()
    
    # Set seed
    set_seed(args.seed)
    
    # Loads config file for fixed configs
    f_config = open('configs/config_legacy_for_sweep.yaml','rb')
    legacy_cfg = yaml.load(f_config, Loader=yaml.FullLoader)
    legacy_cfg['MODEL']['n_classes'] = len(args.classes_to_use)
    
    # legacy_cfg.update(sweeping_config)
    # print(f'Updated config: {legacy_cfg}')
    legacy_cfg['SOLVER']['lr'] = sweeping_config['lr']
    legacy_cfg['SOLVER']['optimizer'] = sweeping_config['optimizer']
    legacy_cfg['SOLVER']['weight_decay'] = sweeping_config['weight_decay']
    legacy_cfg['DATALOADER']['BATCH_SIZE'] = sweeping_config['batch_size']
    legacy_cfg['TRAINING']['EPOCHS'] = sweeping_config['epochs']
    legacy_cfg['TRANSFORMS']['spacing'] = sweeping_config['spacing']
    print(f'Updated config: {legacy_cfg}')
    
    os.environ["CUDA_DEVICE_ORDER"]=legacy_cfg['TRAINING']['CUDA_DEVICE_ORDER']
    df = pd.read_csv(replace_data_path(legacy_cfg[args.dataset]['labelsroot']))
    df = df[df['Group'].isin(args.classes_to_use)]
    
    kfold_results = {"val_accs": [],
                     "best_epoch": [],
                     "recalls": [],
                     "f1s": [],
                     "corrects": [],
                     "n_datapoints": [],
                     "ratios": []}

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.seed)
    
    for i, (train_index, test_index) in enumerate(skf.split(df, df['Group'])):        
        
        # Monai logs foldernames
        legacy_cfg['TRANSFORMS']['cache_dir_train'] = f'./monai_logs/train_sweep_{RUN_NAME}'
        legacy_cfg['TRANSFORMS']['cache_dir_test'] = f'./monai_logs/test_sweep_{RUN_NAME}'
        
        ### DATA ###
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]

        train_dataloader, test_dataloader, train_dataset, test_dataset, ratios_train, ratios_test = make_kfold_dataloaders(legacy_cfg, args, df_train, df_test, verbose=False)

        del df_train, df_test

        ### MODEL ####
        model = make_model(legacy_cfg, args)
        model = prepare_model_for_training(model, legacy_cfg, verbose=False)
        
        # Initialize loss function (with weight balance)
        class_numbers = []
        for class_name in args.classes_to_use:
            class_numbers += [args.classes_to_use.index(class_name)] * ratios_train[class_name]
        class_weights = torch.Tensor(compute_class_weight(class_weight='balanced', classes=np.unique(class_numbers), y=class_numbers)).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        lit_vit = LitViT(model, legacy_cfg['MODEL']['n_classes'],
                         pretrained_model_path=args.use_pretrained,
                         loss_fn=criterion, 
                         optimizer=legacy_cfg['SOLVER']['optimizer'],
                         learning_rate=legacy_cfg['SOLVER']['lr'],
                         weight_decay=legacy_cfg['SOLVER']['weight_decay'], 
                         betas=(legacy_cfg['SOLVER']['beta1'], legacy_cfg['SOLVER']['beta2']),
                         epochs=legacy_cfg['TRAINING']['EPOCHS'],
                         mode=legacy_cfg['MODE'])
                
        trainer = L.Trainer(max_epochs=legacy_cfg['TRAINING']['EPOCHS'],
                            accelerator='gpu',
                            devices=[0],
                            num_sanity_val_steps=0,
                            # logger=None,
                            # callbacks=[checkpoint_callback],
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
        
        # logger.info(kfold_results)
        shutil.rmtree(f'./monai_logs/train_sweep_{RUN_NAME}')
        shutil.rmtree(f'./monai_logs/test_sweep_{RUN_NAME}')

        del model, train_dataloader, test_dataloader, train_dataset, test_dataset
        
    # save kfold results acc into a file
    with open(f'./results/kfold_results_{RUN_NAME}.txt', 'w') as f:
        f.write(f'{RUN_NAME}\n')
        f.write(str(kfold_results) + '\n')
        f.write(f'k-fold acc: {sum(kfold_results["corrects"])/sum(kfold_results["n_datapoints"]):.2f}')
        f.write(f'avg of val accs {round(sum(kfold_results["val_accs"])/4, 2)} ± {round(np.std(kfold_results["val_accs"]), 2)}')
        f.write(f'avg of recalls {round(sum(kfold_results["recalls"])/4, 2)} ± {round(np.std(kfold_results["recalls"]), 2)}')
        f.write(f'avg of f1s {round(sum(kfold_results["f1s"])/4, 2)} ± {round(np.std(kfold_results["f1s"]), 2)}\n\n')
        
    return kfold_results
    
def train_sweep():
    
    with wandb.init(project="DAMIT-Sweep") as run:
        sweeping_config = run.config
        
        kfold_results = train_kfold(sweeping_config)
        
        # after all folds are done, print the results
        kfold_acc = sum(kfold_results["corrects"]) / sum(kfold_results["n_datapoints"])
        kfold_acc_std = np.std(kfold_results["val_accs"])
        recall = sum(kfold_results["recalls"]) / 4
        recall_std = np.std(kfold_results["recalls"])
        f1 = sum(kfold_results["f1s"]) / 4
        f1_std = np.std(kfold_results["f1s"])
        
        run.log({
            "val_acc": kfold_acc,
            "val_acc_std": kfold_acc_std,
            "recall": recall,
            "recall_std": recall_std,
            "f1": f1,
            "f1_std": f1_std
        })

if __name__ == '__main__':
    sweep_config = {
        "method": "grid",
        "metric": {"name": "val_acc", "goal": "maximize"},
        "parameters": {
            "lr": {"values": [1e-5, 1e-4, 1e-3, 1e-2, 0.1]},
            "batch_size": {"values": [4, 8]},
            "epochs": {"values": [50, 100]},
            "optimizer": {"values": ["AdamW", "SGD", "Adam"]},
            "weight_decay": {"values": [0.0, 1e-4, 1e-3, 1e-2, 1e-1]},
            "spacing": {"values": [[1.75, 1.75, 1.75], [1, 1, 1]]}
        }
    }
    
    sweep_id = 'oc05fuhm'
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep_config, project="DAMIT-Sweep")

    print(f"Sweep ID: {sweep_id}")
    wandb.agent(sweep_id, function=train_sweep, project="DAMIT-Sweep")
    
    