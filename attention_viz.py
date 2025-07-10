import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import yaml
import random
import monai
from natsort import natsorted
from tqdm import tqdm
from loguru import logger

import torch

from models.vit3d import Vision_Transformer3D

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    logger.info('Seed is set.')

def make_dataset(cfg, dataset, test_df):
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

    nii_list = natsorted(glob.glob(cfg[dataset]['dataroot'] + '*/hdbet_*[!mask].nii.gz'))
    logger.info(f'{len(nii_list)} nii files found.')
    
    test_datalist = []
    classes_to_use = ['CN', 'AD']
    
    for _, row in test_df.iterrows():
        label = classes_to_use.index(row["Group"])
        path_to_file = [x for x in nii_list if row['Subject'] in x and row['Image Data ID'] in x]
        assert len(path_to_file) == 1, f'More than one file found for {row["Subject"]} and {row["Image Data ID"]}'

        test_datalist.append({
            "image": path_to_file[0],
            "label": label
        })

    test_dataset = monai.data.Dataset(data=test_datalist, 
                                          transform=test_transforms)
    logger.info(f'Test dataset len: {len(test_dataset)}')
    
    return test_dataset

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Attention Visualization')
    parser.add_argument('--config_file', type=str, default='configs/vit3d.yaml', help='Path to the config file')
    parser.add_argument('--dataset', type=str, default='ADNI2', choices=['ADNI1', 'ADNI2'], help='Dataset to use for visualization')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--classes_to_use', nargs='+', type=str, help='Classes to use (enter by separating by space, e.g. CN AD MCI)')
    parser.add_argument('--ft_savename', type=str, default='./models/vit3d.pth', help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./attention_viz_results/', help='Directory to save attention maps')
    parser.add_argument('--view', type=str, default='sagittal', help='MRI view to visualize (sagittal, coronal, axial)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    return parser.parse_args()

def visualize_attention_map(img_rgb, heatmap, overlay, subject_id, slice_id, gt_label, pred_label, output_dir, view='sagittal'):
    """
    Visualize the attention map on top of the image slice and save it.
    
    Parameters
    ----------
    image_slice : np.ndarray
        The image slice to visualize.
    mask_slice : np.ndarray
        The attention map to visualize.
    output_dir : str
        Directory to save the attention map.
    subject_id : str
        Subject ID for naming the output file.
    """
    
    # Display
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb[..., ::-1])  # Convert BGR to RGB
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'image_v-{view}_sub-{subject_id}_slice-{slice_id}_label-{gt_label}_pred-{pred_label}.png'))
    plt.close()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay[..., ::-1])  # Convert BGR to RGB
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'overlay_v-{view}_sub-{subject_id}_slice-{slice_id}_label-{gt_label}_pred-{pred_label}.png'))
    plt.close()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap[..., ::-1], cmap='jet')  # Convert BGR to RGB
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'heatmap_v-{view}_sub-{subject_id}_slice-{slice_id}_label-{gt_label}_pred-{pred_label}.png'))
    plt.close()

def main():
    # python attention_viz.py --config_file configs/config_vitb.yaml --dataset ADNI2 --seed 5426 --classes_to_use CN AD --ft_savename vitb_cnad_04 --view coronal
    args = get_args()
    
    if args.view == 'sagittal':
        view_permute = [0, 1, 2, 3]
    elif args.view == 'coronal':
        view_permute = [0, 2, 1, 3]
    elif args.view == 'axial':
        view_permute = [0, 3, 1, 2]
    
    # Loads config file for fixed configs
    f_config = open(args.config_file,'rb')
    cfg = yaml.load(f_config, Loader=yaml.FullLoader)
    cfg['MODEL']['n_classes'] = len(args.classes_to_use)

    set_seed(args.seed)
    
    # load the dataset
    df = pd.read_csv(cfg[args.dataset]['labelsroot'])
    df = df[df['Group'].isin(args.classes_to_use)]
    
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.seed)
    
    POSTFIX = f"{args.ft_savename}_{args.dataset}_seed_{args.seed}"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/{POSTFIX}", exist_ok=True)

    for i, (train_index, test_index) in enumerate(skf.split(df, df['Group'])):
        os.makedirs(f"{args.output_dir}/{POSTFIX}/fold_{i}", exist_ok=True)
        logger.info(f'Fold {i}')
        test_df = df.iloc[test_index]

        test_dataset = make_dataset(cfg, args.dataset, test_df)
        print('Len of test dataset:', len(test_dataset))
        
        ft_checkpoint = glob.glob(f"./checkpoints/BEST_MODEL_{POSTFIX}_fold_{i}*.pth.tar")
        assert len(ft_checkpoint) == 1, f'More than one checkpoint found for fold {i}: {ft_checkpoint}'
        
        # initialize the model
        vit_model = Vision_Transformer3D(**cfg['MODEL'])
        checkpoint = torch.load(ft_checkpoint[0], map_location='cpu')
        checkpoint_model = checkpoint['net']
        msg = vit_model.load_state_dict(checkpoint_model, strict=True)
        logger.warning(f"Missing keys: {msg.missing_keys}")
        logger.warning(f"Loading checkpoint from {ft_checkpoint}")
        
        vit_model.to(args.gpu)
        vit_model.eval()
        
        sub_id = 0
        for sample in tqdm(test_dataset):
            sub_id += 1
            with torch.no_grad():
                sample_label = sample['label'].to(args.gpu, non_blocking=True).item()
                gt_label = args.classes_to_use[sample_label]
                
                pred, pred_attn = vit_model(sample['image'].unsqueeze(0).to(args.gpu, non_blocking=True), return_attn=True)
                pred = torch.argmax(pred, dim=1).cpu().item()
                pred_label = args.classes_to_use[pred]
                
                pred_attn = pred_attn.mean(dim=2).squeeze(0) # average over heads shape: [num_layers, num_heads, num_patches, num_patches] --> [num_layers, num_patches, num_patches]

                num_layers = pred_attn.shape[0]
                eye = torch.eye(pred_attn[0].size(-1)).to(args.gpu)
                pred_attn = [(att + eye) / 2 for att in pred_attn] # add identity to attention maps shape: [num_layers, num_patches, num_patches]

                # Multiply matrices
                joint_attention = pred_attn[0] # start with the first layer
                for l in range(1, num_layers): 
                    joint_attention = pred_attn[l] @ joint_attention # matrix multiplication shape : [num_patches, num_patches] @ [num_patches, num_patches] = [num_patches, num_patches]
                
                cls_attention = joint_attention[0, 1:]  # skip CLS token itself
                # print('CLS token attention shape:', cls_attention.shape)
                
                num_patches = cls_attention.shape[0]
                grid_size = int(np.cbrt(num_patches))
                mask = cls_attention.reshape(grid_size, grid_size, grid_size)
            
            mask = mask.unsqueeze(0).unsqueeze(0)
            
            resized_attn = torch.nn.functional.interpolate(mask, size=[128, 128, 128], mode='trilinear').squeeze(0)
            resized_attn = resized_attn.permute(view_permute).cpu().numpy()
            
            slice_id = random.randint(40, 80)
            sample['image'] = sample['image'].permute(view_permute)
            image_slice = sample['image'].cpu().numpy()[0, slice_id, :, :]

            # Normalize image to [0, 255]
            img_norm = cv2.normalize(image_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

            # Normalize and colorize attention mask
            mask_slice = resized_attn[0, slice_id, :, :]
            attn_norm = (mask_slice - mask_slice.min()) / (mask_slice.max() - mask_slice.min())
            heatmap = cv2.applyColorMap(np.uint8(255 * attn_norm), cv2.COLORMAP_JET)
            
            # Overlay
            alpha = 0.5
            overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
        
            # Save attention maps
            visualize_attention_map(img_rgb=img_rgb, 
                                heatmap=heatmap, 
                                overlay=overlay, 
                                subject_id=sub_id, 
                                slice_id=slice_id, 
                                gt_label=gt_label, 
                                pred_label=pred_label,
                                output_dir=f"{args.output_dir}/{POSTFIX}/fold_{i}/",
                                view=args.view)
    
    logger.info('Attention maps saved successfully.')
        
if __name__ == "__main__":
    main()
        
                