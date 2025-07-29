import os
import glob
import pandas as pd
import monai

import torch

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, cfg, n_views=2):
        self.base_transform = monai.transforms.Compose([monai.transforms.LoadImaged(keys=["image"]),
                                                        monai.transforms.EnsureChannelFirstd(keys=["image"]),
                                                        monai.transforms.Orientationd(keys=["image"], axcodes="RAS"),
                                                        monai.transforms.Spacingd(keys=["image"], pixdim=(1.75, 1.75, 1.75)),
                                                        monai.transforms.RandSpatialCropd(keys=["image"], roi_size=(10, 10, 10),
                                                                                        max_roi_size=(128, 128, 128), random_center=True,
                                                                                        random_size=True),
                                                        monai.transforms.Resized(keys=["image"], spatial_size=(128, 128, 128), mode="trilinear"),
                                                        monai.transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                                                        monai.transforms.RandShiftIntensityd(keys=["image"], prob=0.8, offsets=0.1),
                                                        monai.transforms.RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7, 1.3)),
                                                        monai.transforms.RandGaussianSmoothd(keys=["image"], sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5), prob=0.8)
                                            ])
        self.n_views = n_views

    def __call__(self, x):
        return tuple(self.base_transform(x)['image'] for i in range(self.n_views))

def check_dirs_exist(cfg):
    if not os.path.exists(cfg["BRATS2023"]["dataroot"]):
        cfg["BRATS2023"]["dataroot"] = cfg["BRATS2023"]["dataroot"].replace("/SSD/", "/SSD/guest/")
    if not os.path.exists(cfg["IXI"]["dataroot"]):
        cfg["IXI"]["dataroot"] = cfg["IXI"]["dataroot"].replace("/SSD/", "/SSD/guest/")
    if not os.path.exists(cfg["HCP"]["dataroot"]):
        cfg["HCP"]["dataroot"] = cfg["HCP"]["dataroot"].replace("/SSD/", "/SSD/guest/")
    if not os.path.exists(cfg["OASIS3"]["dataroot"]):
        cfg["OASIS3"]["dataroot"] = cfg["OASIS3"]["dataroot"].replace("/SSD/", "/SSD/guest/")
    if not os.path.exists(cfg["OASIS3"]["labelsroot"]):
        cfg["OASIS3"]["labelsroot"] = cfg["OASIS3"]["labelsroot"].replace("/SSD/", "/SSD/guest/")
    
    return cfg

def get_dataset_list(datasets, cfg):
    cfg = check_dirs_exist(cfg)
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

def contrastive_collate_fn(batch):
    """
    Custom collate function to handle batches of images for contrastive learning.
    This function assumes that each item in the batch is a list of crops.
    """
    # Unzip the batch into individual crops
    crops = list(zip(*batch)) # [ (view1, view2 (view1, view2) ]
    # Stack each crop type into a tensor
    crops_stacked = [torch.stack(crop) for crop in crops] # [ (view1, ...), (view2, ...) ]
    return crops_stacked
                                                                               
def make_monai_dataset_for_simclr(datasets, cfg):
    
    n_views = cfg['simclr']['n_views']
    
    dataset_list = get_dataset_list(datasets, cfg)
    
    transforms_n_views = ContrastiveLearningViewGenerator(cfg=cfg, n_views=n_views)
    
    dataset = monai.data.PersistentDataset(
        data=dataset_list,
        transform=transforms_n_views,
        cache_dir=cfg['cache_dir']
    )
        
    return dataset
