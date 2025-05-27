import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import yaml
import random
import monai
from natsort import natsorted
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import roc_curve, auc

import torch

from models.vit3d_cls import Vision_Transformer3D
# from utils.utils import load_pretrained_checkpoint
from sklearn.manifold import TSNE
import seaborn as sns

# Assuming:
# - tokens is a NumPy array of shape [N, 768] (N = number of samples)
# - labels is a list or array of length N with integer or string class labels
def plot_tsne(tokens, labels, perplexity=30, random_state=42, fig_savename=None):
    logger.info("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, init='random', random_state=random_state)
    tokens_2d = tsne.fit_transform(tokens)

    # Create a scatterplot with hue as the class labels
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        x=tokens_2d[:, 0], 
        y=tokens_2d[:, 1],
        hue=labels,
        palette='tab10',
        alpha=0.7,
        edgecolor='k',
        linewidth=0.2
    )
    # plt.title('t-SNE of CLS Tokens')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    plt.legend(title='Class', bbox_to_anchor=(1., 1), loc='upper left')
    plt.tight_layout()
    # plt.grid(True)
    # plt.show()
    plt.savefig(f'./tsne_roc_results/{fig_savename}', dpi=300)
    logger.success(f"t-SNE plot saved to ./tsne_results/{fig_savename}")

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

def plot_roc_curve(y_true, scores, fig_savename=None):
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    # Plot the AUROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('AUROC Curve with Random Values')
    plt.legend(loc='lower right')
    # plt.grid(True)
    plt.savefig(f'./tsne_roc_results/{fig_savename}', dpi=300)
    logger.success(f"ROC curve saved to ./roc_results/{fig_savename}")
    
def predict_and_make_tsne(cfg, savename, dataset, seed, classes_to_use=['CN', 'AD']):
    # Set seed
    set_seed(seed)
    cfg['MODEL']['n_classes'] = 2
    
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)

    df = pd.read_csv(cfg[dataset]['labelsroot'])
    df = df[df['Group'].isin(classes_to_use)]
    
    cls_token_labels = []
    labels = []
    preds = []
    confidences = []

    for i, (train_index, test_index) in enumerate(skf.split(df, df['Group'])):
        logger.info(f'Fold {i}')
        
        test_df = df.iloc[test_index]

        test_dataset = make_dataset(cfg, dataset, test_df)
        logger.info('Len of test dataset:', len(test_dataset))
        
        # initialize the model
        vit_model = Vision_Transformer3D(**cfg['MODEL'])
        # bl = vit_model.blocks[0].attn.qkv.weight
        # print(bl[0, :10])

        ft_checkpoint = glob.glob(f'./checkpoints/BEST_MODEL_{savename}_{dataset}_seed_{seed}_fold_{i}*pth.tar')[0]
        logger.warning(f'Loading checkpoint from {ft_checkpoint}')
        # load the checkpoint
        checkpoint = torch.load(ft_checkpoint, map_location='cpu')
        checkpoint_model = checkpoint['net']
        msg = vit_model.load_state_dict(checkpoint_model, strict=True)
        logger.info(f'Missing keys: {msg.missing_keys}')

        vit_model.to(7)
        vit_model.eval()
        for sample in tqdm(test_dataset):
            pred, cls_token = vit_model(sample['image'].unsqueeze(0).to(7), return_cls=True)
            # print(sample['label'], torch.argmax(pred, dim=1))
            labels.append(sample['label'].cpu().numpy())
            # confidences: prob scores of the class 1
            confidences.append(torch.softmax(pred, dim=1)[:, 1].item())
            preds.append(torch.argmax(pred, dim=1))
            cls_token_labels.append({'cls_token': cls_token.squeeze(0).detach().cpu().numpy(),
                                    'label': sample['label'].cpu().numpy()})

    cls_tokens_arr = np.array([x['cls_token'] for x in cls_token_labels])
    cls_tokens_labels_arr = np.array([x['label'] for x in cls_token_labels])
    
    # calc acc
    y_hat = [p.item() for p in preds]
    y_true = [l.item() for l in labels]
    acc = np.sum(np.array(y_hat) == np.array(y_true)) / len(y_true)
    logger.info(f'Accuracy: {acc:.4f}, while Len: {len(y_true)} samples')

    fig_savename = f'tsne_{savename}_d_{dataset}_s_{seed}.png'
    plot_tsne(cls_tokens_arr, cls_tokens_labels_arr, perplexity=30, random_state=seed, fig_savename=fig_savename)
    
    fig_savename = f'roc_{savename}_d_{dataset}_s_{seed}.png'
    plot_roc_curve(y_true, confidences, fig_savename=fig_savename)

# def get_args():
#     import argparse
#     parser = argparse.ArgumentParser(description='Predict and make t-SNE plot')
#     parser.add_argument('--cfg', type=str, default='./configs/vit3d_cls.yaml', help='Path to the config file')
#     parser.add_argument('--savename', type=str, default='vitb_cnad_04', help='Name of the model')
#     parser.add_argument('--dataset', type=str, default='ADNI', help='Dataset name')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed')
#     args = parser.parse_args()
#     return args

if __name__ == "__main__":
    
    savenames = ['vitb_cnad_04', 'linear_cnad_02', 'tfs_cnad_01']
    datasets = ['ADNI1', 'ADNI2']
    classes_to_use = ['CN', 'AD']
    config_file = 'configs/config_vitb.yaml'
    
    for savename in savenames:
        for dataset in datasets:
            cfg = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
            seed = 3148 if dataset == 'ADNI2' else 7661
            logger.warning(f'Running t-SNE adn ROC curve for {savename} on {dataset} with seed {seed}')
            predict_and_make_tsne(cfg, savename, dataset, seed, classes_to_use)
            logger.success(f'Finished t-SNE and ROC curve for {savename} on {dataset} with seed {seed}')