# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 2023

@author: qasymjomart
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import os
from natsort import natsorted
import monai
from monai import data

def replace_data_path(datapath):
    if not os.path.exists(datapath):
        datapath = datapath.replace('/SSD/qasymjomart/', '/SSD/guest/qasymjomart/')
    return datapath


def make_train_test_split_of_target(cfg, args, test_size=0.2):
    """
    Make train and test split of the target dataset.

    Parameters:
    ------------
    cfg: config (read from yaml)
    args: argument parser from command line
    test_split: test split percentage (in decimal)

    Returns:
    ---------
    train_test_split_paths : dict(train_path, test_path)

    """
    if args.dataset in ['ADNI1', 'ADNI2']:
        labels = pd.read_csv(cfg[args.dataset]['labelsroot'])
        x_train, x_test, y_train, y_test = train_test_split(labels.Subject, 
                                                            labels.Group, 
                                                            test_size=test_size, 
                                                            stratify=labels.Group
                                                            )
        labels_test = pd.concat([x_test, y_test], axis=1)
        labels_test.sort_values(by='Subject', inplace=True)
        test_csv = pd.merge(labels_test, 
                            labels[['Subject', 'Image Data ID']], 
                            on=['Subject'], 
                            how='left'
                            )
        
        labels_train = pd.concat([x_train, y_train], axis=1)
        labels_train.sort_values(by='Subject', inplace=True)
        train_csv = pd.merge(labels_train, 
                             labels[['Subject', 'Image Data ID']], 
                             on=['Subject'], 
                             how='left'
                            )
        
        test_csv.to_csv(cfg['DATALOADER']['TRAIN_TEST_SPLIT_PATH'] + 'test_' + args.savename + "_" + args.mode + '_seed_' + str(args.seed) + '.csv')
        train_csv.to_csv(cfg['DATALOADER']['TRAIN_TEST_SPLIT_PATH'] + 'train_' + args.savename + "_" + args.mode + '_seed_' + str(args.seed) + '.csv')

        train_test_split_paths = {
            'train' : cfg['DATALOADER']['TRAIN_TEST_SPLIT_PATH'] + 'train_' + args.savename + "_" + args.mode + '_seed_' + str(args.seed) + '.csv',
            'test' : cfg['DATALOADER']['TRAIN_TEST_SPLIT_PATH'] + 'test_' + args.savename + "_" + args.mode + '_seed_' + str(args.seed) + '.csv'
        }

    elif args.dataset in ['OASIS3']:
        labels = pd.read_csv(cfg[args.dataset]['labelsroot'])
        x_train, x_test, y_train, y_test = train_test_split(labels.OASISID, 
                                                            labels.Label, 
                                                            test_size=test_size, 
                                                            stratify=labels.Label
                                                            )
        labels_test = pd.concat([x_test, y_test], axis=1)
        labels_test.sort_values(by='OASISID', inplace=True)
        test_csv = pd.merge(labels_test, 
                            labels[['OASISID', 'MRI_LABEL', 'filename']], 
                            on=['OASISID'], 
                            how='left'
                            )
        
        labels_train = pd.concat([x_train, y_train], axis=1)
        labels_train.sort_values(by='OASISID', inplace=True)
        train_csv = pd.merge(labels_train, 
                             labels[['OASISID', 'MRI_LABEL', 'filename']], 
                             on=['OASISID'], 
                             how='left'
                            )
        
        test_csv.to_csv(cfg['DATALOADER']['TRAIN_TEST_SPLIT_PATH'] + 'test_' + args.savename + "_" + args.mode + '_seed_' + str(args.seed) + '.csv')
        train_csv.to_csv(cfg['DATALOADER']['TRAIN_TEST_SPLIT_PATH'] + 'train_' + args.savename + "_" + args.mode + '_seed_' + str(args.seed) + '.csv')

        train_test_split_paths = {
            'train' : cfg['DATALOADER']['TRAIN_TEST_SPLIT_PATH'] + 'train_' + args.savename + "_" + args.mode + '_seed_' + str(args.seed) + '.csv',
            'test' : cfg['DATALOADER']['TRAIN_TEST_SPLIT_PATH'] + 'test_' + args.savename + "_" + args.mode + '_seed_' + str(args.seed) + '.csv'
        }
    
    return train_test_split_paths

def make_mae_pretraining_dataloaders(cfg, args):
    """Build datalaoders for MAE pretraining

    Args:
        cfg : config (read from yaml)
        args : argument parser from command line
    
    Returns:
    ------------
    pretraining_dataloader : type (torch.utils.data.DataLoader)

    """
    dataset_list = []
    datapath_list = []
    
    if "SYNTH1" in args.datasets:
        datapath_list = datapath_list + glob.glob(cfg["SYNTH1"]["dataroot"])
        print('Used SYNTH1')
    
    if "SYNTH2" in args.datasets:
        datapath_list = datapath_list + glob.glob(cfg["SYNTH2"]["dataroot"])
        print('Used SYNTH2')
    
    if "BRATS2023" in args.datasets:
        datapath_list = datapath_list + glob.glob(cfg["BRATS2023"]["dataroot"])
        print('Used BRATS2023')    

    if "IXI" in args.datasets:
        datapath_list = datapath_list + glob.glob(cfg['IXI']['dataroot'])
    
    if "HCP" in args.datasets:
        datapath_list = datapath_list + glob.glob(cfg['HCP']['dataroot'])
    
    if "ADNI1" in args.datasets:
        datapath_list = datapath_list + glob.glob(cfg['ADNI1']['dataroot'])
    
    if "ADNI2" in args.datasets:
        datapath_list = datapath_list + glob.glob(cfg['ADNI2']['dataroot'])
    
    if "OASIS3" in args.datasets: # have to include code for sorting out healthy subjects
        datapath_temp_list = glob.glob(cfg["OASIS3"]["dataroot"])
        # Step 1: Read the cfg["OASIS3"]["labelsroot"] file into a DataFrame and select the "filename" column
        df_healthy_oasis3 = pd.read_csv(cfg["OASIS3"]["labelsroot"])
        filenames = df_healthy_oasis3['filename'].tolist()
        # Step 2: Remove the .json ending from each entry in the list, replace it with .nii.gz, and add 'hdbet_' as a prefix
        filenames = ['hdbet_' + filename.replace('.json', '.nii.gz') for filename in filenames]
        # Step 3: Select the subset of datapath_list where the filename matches with the entries of the earlier created list
        datapath_list = datapath_list + [path for path in datapath_temp_list if os.path.basename(path) in filenames]

    for data_path in datapath_list:
        dataset_list.append({"image": data_path})
    
    if args.use_aug:
        train_transforms = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image",]),
            monai.transforms.Orientationd(keys=["image"], axcodes=cfg["TRANSFORMS"]["orientation"]),
            monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg["TRANSFORMS"]["spacing"])),
            monai.transforms.NormalizeIntensityd(keys=["image"], nonzero=cfg["TRANSFORMS"]["normalize_non_zero"]),
            # monai.transforms.RandSpatialCropd(keys=["image"], roi_size=(80,80,80), max_roi_size=(150,150,150)),
            monai.transforms.CropForegroundd(keys=["image"], source_key="image"), 
            monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg["TRANSFORMS"]["resize"])),
            monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
            monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
            monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
            monai.transforms.RandRotate90d(keys=["image"], prob=0.2, max_k=3),
            monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            monai.transforms.ToTensord(keys=["image"])
            ])
    else:
        print('No data augmentations used')
        train_transforms = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image",]),
            monai.transforms.Orientationd(keys=["image"], axcodes=cfg["TRANSFORMS"]["orientation"]),
            monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg["TRANSFORMS"]["spacing"])),
            monai.transforms.RandSpatialCropd(keys=["image"], roi_size=(80,80,80), max_roi_size=(150,150,150)),
            monai.transforms.CropForegroundd(keys=["image"], source_key="image"), 
            monai.transforms.NormalizeIntensityd(keys=["image"], nonzero=cfg["TRANSFORMS"]["normalize_non_zero"]),
            monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg["TRANSFORMS"]["resize"])),
            monai.transforms.ToTensord(keys=["image"])
            ])
        
    pretrain_dataset = data.PersistentDataset(
            data=dataset_list,
            transform=train_transforms,
            cache_dir=cfg['TRANSFORMS']['cache_dir_train']
        )
    
    pretraining_dataloader = data.DataLoader(pretrain_dataset, 
                                    batch_size=cfg['DATALOADER']['BATCH_SIZE'],
                                    shuffle=True, 
                                    num_workers=cfg['DATALOADER']['NUM_WORKERS'],
                                    pin_memory=True,
                                    # drop_last=True
                                    )
    
    return pretraining_dataloader, pretrain_dataset


def make_kfold_dataloaders(cfg, args, train_df, test_df):

    if args.use_aug:
        # new augmentation
        # train_transforms = monai.transforms.Compose([
        #     monai.transforms.LoadImaged(keys=["image"]),
        #     monai.transforms.EnsureChannelFirstd(keys=["image"]),
        #     monai.transforms.Orientationd(keys=["image"], axcodes=cfg["TRANSFORMS"]["orientation"]),
        #     monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg["TRANSFORMS"]["spacing"])),
        #     monai.transforms.CropForegroundd(keys=["image"], source_key="image"), 
        #     monai.transforms.NormalizeIntensityd(keys=["image"], nonzero=cfg["TRANSFORMS"]["normalize_non_zero"]),
        #     # monai.transforms.RandZoomd(keys=["image"], prob=0.2, min_zoom=0.9, max_zoom=1.2, keep_size=True),
        #     monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg["TRANSFORMS"]["resize"])),
        #     # monai.transforms.RandAffined(keyss=["image"], prob=0.2, rotate_range=(0, 0, 0.2 * math.pi), scale_range=(0.1, 0.1, 0.1), translate_range=(10, 10, 10)),
        #     # monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
        #     # monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
        #     # monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
        #     # monai.transforms.RandRotate90d(keys=["image"], prob=0.2, max_k=3),
        #     # monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.2), # must be disabled
        #     # monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2), # must be disabled
        #     # monai.transforms.RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1), # must be disabled
        #     monai.transforms.RandScaleIntensityd(keys="image", factors=0.2, prob=0.5),  # Increased factors and probability
        #     monai.transforms.RandShiftIntensityd(keys="image", offsets=0.2, prob=0.5),  # Increased offsets and probability
        #     monai.transforms.RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.2),  # Increased standard deviation and probability
        #     monai.transforms.ToTensord(keys=["image", "label"])
        # ])
        
        # old augmentation
        train_transforms = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
            monai.transforms.Orientationd(keys=["image"], axcodes=cfg["TRANSFORMS"]["orientation"]),
            monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg["TRANSFORMS"]["spacing"])),
            monai.transforms.CropForegroundd(keys=["image"], source_key="image"), 
            monai.transforms.NormalizeIntensityd(keys=["image"], nonzero=cfg["TRANSFORMS"]["normalize_non_zero"]),
            # monai.transforms.RandZoomd(keys=["image"], prob=0.2, min_zoom=0.9, max_zoom=1.2, keep_size=True),
            monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg["TRANSFORMS"]["resize"])),
            # monai.transforms.RandAffined(keyss=["image"], prob=0.2, rotate_range=(0, 0, 0.2 * math.pi), scale_range=(0.1, 0.1, 0.1), translate_range=(10, 10, 10)),
            monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
            monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
            monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
            monai.transforms.RandRotate90d(keys=["image"], prob=0.2, max_k=3),
            monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.2), # must be disabled
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2), # must be disabled
            monai.transforms.RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1), # must be disabled
            monai.transforms.ToTensord(keys=["image", "label"])
        ])
        
    else:
        train_transforms = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
            monai.transforms.Orientationd(keys=["image"], axcodes=cfg["TRANSFORMS"]["orientation"]),
            monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg["TRANSFORMS"]["spacing"])),
            monai.transforms.CropForegroundd(keys=["image"], source_key="image"), 
            monai.transforms.NormalizeIntensityd(keys=["image"], nonzero=cfg["TRANSFORMS"]["normalize_non_zero"]),
            monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg["TRANSFORMS"]["resize"])),
            monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),
            monai.transforms.ToTensord(keys=["image", "label"])
        ])

    test_transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image",]),
        monai.transforms.Orientationd(keys=["image"], axcodes=cfg["TRANSFORMS"]["orientation"]),
        monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg["TRANSFORMS"]["spacing"])),
        monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
        monai.transforms.NormalizeIntensityd(keys=["image"], nonzero=cfg["TRANSFORMS"]["normalize_non_zero"]),
        monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg["TRANSFORMS"]["resize"])),
        monai.transforms.ToTensord(keys=["image", "label"])
    ])

    nii_list = natsorted(glob.glob(replace_data_path(cfg[args.dataset]['dataroot']) + '*/hdbet_*[!mask].nii.gz'))
    print(f'{len(nii_list)} nii files found.')

    # if need to train with few samples, split in a stratified fashion
    if cfg["DATALOADER"]["train_size"] in [0.1, 0.2, 0.4, 0.6, 0.8]:
        train_df, _, _, _ = train_test_split(train_df, train_df["Group"], 
                                             train_size=int(cfg["DATALOADER"]["train_size"]*len(train_df)), random_state=args.seed,
                                             shuffle=True, stratify=train_df["Group"])
        print(f'Few sample training of {100*cfg["DATALOADER"]["train_size"]} % samples: {len(train_df)}')

    train_datalist = []
    for _, row in train_df.iterrows():
        label = args.classes_to_use.index(row["Group"])
        path_to_file = [x for x in nii_list if row['Subject'] in x and row['Image Data ID'] in x]
        assert len(path_to_file) == 1, f'More than one file found for {row["Subject"]} and {row["Image Data ID"]}'
        
        ###
        # label noise injection
        # if cfg['DATALOADER']['label_noise'] is not None:
            # if random.random() < cfg['DATALOADER']['label_noise']:
                # label = 1 - label
        ###
        
        train_datalist.append({
            "image": path_to_file[0],
            "label": label
        })

    ratios_train = {}
    for label in args.classes_to_use:
        label_id = args.classes_to_use.index(label)
        ratios_train[label] = sum([1 for x in train_datalist if x['label'] == label_id])
    
    train_dataset = data.PersistentDataset(data=train_datalist, 
                                           transform=train_transforms, 
                                           cache_dir=cfg['TRANSFORMS']['cache_dir_train'])
    print(f'Train dataset len: {len(train_dataset)}')
    
    test_datalist = []

    for _, row in test_df.iterrows():
        label = args.classes_to_use.index(row["Group"])
        path_to_file = [x for x in nii_list if row['Subject'] in x and row['Image Data ID'] in x]
        assert len(path_to_file) == 1, f'More than one file found for {row["Subject"]} and {row["Image Data ID"]}'

        test_datalist.append({
            "image": path_to_file[0],
            "label": label
        })

    test_dataset = data.PersistentDataset(data=test_datalist, 
                                          transform=test_transforms, 
                                          cache_dir=cfg['TRANSFORMS']['cache_dir_test'])
    print(f'Test dataset len: {len(test_dataset)}')

    train_dataloader = data.DataLoader(train_dataset, 
                        batch_size=cfg['DATALOADER']['BATCH_SIZE'],
                        shuffle=True, 
                        num_workers=cfg["DATALOADER"]["NUM_WORKERS"],
                        drop_last=True,
                        pin_memory=True
                        )

    test_dataloader = data.DataLoader(test_dataset, 
                                    batch_size=1,
                                    shuffle=False, 
                                    num_workers=0
                                    )

    ratios_test = {}
    for label in args.classes_to_use:
        label_id = args.classes_to_use.index(label)
        ratios_test[label] = sum([1 for x in test_datalist if x['label'] == label_id])

    return train_dataloader, test_dataloader, train_dataset, test_dataset, ratios_train, ratios_test