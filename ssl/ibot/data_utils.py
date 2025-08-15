# This file contains the data augmentation pipeline for DINO training.
# Specifically tailored for 3D MRI using MONAI datasets.
import glob
import torch
import monai
import pandas as pd
import os

from loader import ImageFolderMask

class Solarizationd(monai.transforms.MapTransform):
    def __init__(self, keys, prob=0.5, threshold=0.5):
        super().__init__(keys)
        self.prob = prob
        self.threshold = threshold

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if torch.rand(1).item() < self.prob:
                x = d[key]
                d[key] = torch.where(x >= self.threshold, 1.0 - x, x)
        return d

class MONAIDataAugmentationiBOT:
    def __init__(self, 
                 resize=[128, 128, 128],
                 orientation="RAS",
                 spacing=[1.75, 1.75, 1.75],
                 local_crop_img_size=[54, 54, 54],
                 global_crops_scale=(0.4, 1.), 
                 local_crops_scale=(0.05, 0.4),
                 global_crops_number=2,
                 local_crops_number=8,
                 **kwargs):
        self.local_crops_number = local_crops_number
        # min_size = tuple([int(img_size * global_crops_scale[0])] * 3)
        max_size = tuple([int(resize[i] * global_crops_scale[1]) for i in range(len(resize))])

        first_trans = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image",]),
            monai.transforms.Orientationd(keys=["image"], axcodes=orientation),
            monai.transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True),
            monai.transforms.Spacingd(keys=["image"], pixdim=tuple(spacing)),
            # monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
            # monai.transforms.Resized(keys=["image"], spatial_size=tuple(resize)),
        ])
        
        rand_trans = monai.transforms.Compose([
            monai.transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            monai.transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            monai.transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            monai.transforms.RandAdjustContrastd(keys=["image"], prob=0.8, gamma=(0.7, 1.3)),
            monai.transforms.RandBiasFieldd(keys=["image"], prob=0.2),
            monai.transforms.RandShiftIntensityd(keys=["image"], prob=0.5, offsets=0.1),
            monai.transforms.RandScaleIntensityd(keys=["image"], prob=0.5, factors=0.2),
        ])

        # norm_trans = monai.transforms.Compose([
        #     monai.transforms.NormalizeIntensityd(keys=["image"], nonzero=True),
        # ])
        
        self.global_crops_number = global_crops_number
        self.global_trans1 = monai.transforms.Compose([
            first_trans,
            monai.transforms.RandScaleCropd(keys=["image"], roi_scale=global_crops_scale[0], max_roi_scale=global_crops_scale[1],
                                            random_center=True, random_size=True),
            monai.transforms.Resized(keys=["image"], spatial_size=max_size, mode="trilinear"),
            rand_trans,
            monai.transforms.GaussianSmoothd(keys=["image"], sigma=1.0),
            # monai.transforms.ScaleIntensityd(keys=["image"]),
            # norm_trans
        ])

        self.global_trans2 = monai.transforms.Compose([
            first_trans,
            # monai.transforms.RandSpatialCropd(keys=["image"], roi_size=roi_size, max_roi_size=max_roi_size, 
                                            #   random_center=True, random_size=True, prob=1.0),
            monai.transforms.RandScaleCropd(keys=["image"], roi_scale=global_crops_scale[0], max_roi_scale=global_crops_scale[1],
                                            random_center=True, random_size=True),
            monai.transforms.Resized(keys=["image"], spatial_size=max_size, mode="trilinear"),
            rand_trans,
            monai.transforms.GaussianSmoothd(keys=["image"], sigma=1.0),
            # monai.transforms.ScaleIntensityd(keys=["image"]),
            # Solarizationd(keys=["image"], prob=0.2, threshold=0.5),
            # norm_trans
        ])

        self.local_trans = monai.transforms.Compose([
            first_trans,
            monai.transforms.RandScaleCropd(keys=["image"], roi_scale=local_crops_scale[0], max_roi_scale=local_crops_scale[1],
                                            random_center=True, random_size=True),
            monai.transforms.Resized(keys=["image"], spatial_size=tuple(local_crop_img_size), mode="trilinear"),
            rand_trans,
            monai.transforms.GaussianSmoothd(keys=["image"], sigma=1.0),
            # monai.transforms.ScaleIntensityd(keys=["image"]),
            # norm_trans
        ])
       

    def __call__(self, image):
        crops = []
        crops.append(self.global_trans1(image)['image'])
        # crops.append(self.global_trans2(image)['image'])
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_trans2(image)['image'])
        for _ in range(self.local_crops_number):
            crops.append(self.local_trans(image)['image'])
        return crops

def get_dataset_list(datasets, cfg):
    datapath_list = []
    if "BRATS2023" in datasets:
        datapath_list = datapath_list + glob.glob(cfg["BRATS2023"]["dataroot"])
        print('Used BRATS2023')    

    if "IXI" in datasets:
        datapath_list = datapath_list + glob.glob(cfg['IXI']['dataroot'])
    
    if "HCP" in datasets:
        datapath_list = datapath_list + glob.glob(cfg['HCP']['dataroot'])
    
    if "OASIS3" in datasets: # have to include code for sorting out healthy subjects
        datapath_temp_list = glob.glob(cfg["OASIS3"]["dataroot"])
        # Step 1: Read the cfg["OASIS3"]["labelsroot"] file into a DataFrame and select the "filename" column
        df_healthy_oasis3 = pd.read_csv(cfg["OASIS3"]["labelsroot"])
        filenames = df_healthy_oasis3['filename'].tolist()
        # Step 2: Remove the .json ending from each entry in the list, replace it with .nii.gz, and add 'hdbet_' as a prefix
        filenames = ['hdbet_' + filename.replace('.json', '.nii.gz') for filename in filenames]
        # Step 3: Select the subset of datapath_list where the filename matches with the entries of the earlier created list
        datapath_list = datapath_list + [path for path in datapath_temp_list if os.path.basename(path) in filenames]

    dataset_list = []
    for data_path in datapath_list:
        dataset_list.append({"image": data_path})
    print(f"Total number of images in dataset: {len(dataset_list)}")

    return dataset_list

def make_ibot_dataloaders(cfg, datasets):
    transform = MONAIDataAugmentationiBOT(
        **cfg['transforms']
    )
    
    dataset_list = get_dataset_list(datasets, cfg)
    print(f"Dataset list length: {len(dataset_list)}")
    
    # dataset = monai.data.PersistentDataset(
    #     data=dataset_list,
    #     transform=transform,
    #     cache_dir=cfg['cache_dir']
    # )
    
    dataset = ImageFolderMask(
        data=dataset_list, 
        transform=transform,
        cache_dir=cfg['cache_dir'],
        patch_size=cfg['model']['patch_size'],
        pred_ratio=cfg['ibot_model']['pred_ratio'],
        pred_ratio_var=cfg['ibot_model']['pred_ratio_var'],
        pred_aspect_ratio=(0.3, 1/0.3),
        pred_shape=cfg['ibot_model']['pred_shape'],
        pred_start_epoch=cfg['ibot_model']['pred_start_epoch'])
    
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=cfg['training']['batch_size'],
    #     num_workers=cfg['data']['num_workers'],
    #     shuffle=True,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg['training']['batch_size_per_gpu'],
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    return dataset, sampler, dataloader