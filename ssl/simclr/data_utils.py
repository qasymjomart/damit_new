import os
import glob
import pandas as pd
import monai

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
                                                        monai.transforms.RandomAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7, 1.3)),
                                                        monai.transforms.RandGaussianSmoothd(keys=["image"], sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5), prob=0.8)
                                            ])
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]
                 
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
                                                    
                                                                               
def make_monai_dataset_for_simclr(datasets, n_views, cfg):
    
    dataset_list = get_dataset_list(datasets, cfg)
    
    transforms_n_views = ContrastiveLearningViewGenerator(cfg=cfg, n_views=n_views)
    
    dataset = monai.data.PersistentDataset(
        data=dataset_list,
        transform=transforms_n_views,
        cache_dir=cfg['cache_dir']
    )

    return dataset
