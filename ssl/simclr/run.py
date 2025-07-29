import os
import yaml
import argparse
import torch
import wandb

from data_utils import make_monai_dataset_for_simclr, contrastive_collate_fn
from models.vit3d_simclr import ViT3DSimCLR
from simclr import SimCLR

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--config_file', type=str, default='configs.yaml', help='Name of the config file')
parser.add_argument('--savename', type=str, help='Experiment name (used for saving files)')
parser.add_argument('--datasets', type=str, nargs='+', default=['IXI'], help='List of datasets to use for training')
parser.add_argument('--seed', default=42, type=int, help='seed for initializing training. ')
parser.add_argument('--devices', type=str, default="0", help='GPU devices to use')
parser.add_argument('--output_dir', type=str, default='./checkpoints/', help='Directory to save output files')

def main():
    args = parser.parse_args()
    # check if gpu training is available
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    f_config = open(args.config_file,'rb')
    cfg = yaml.load(f_config, Loader=yaml.FullLoader)
    
    assert cfg['simclr']['n_views'] == 2, "Only two view training is supported. Please use --n-views 2."
    
    FILENAME = f"SimCLR_pt_{args.savename}_{'_'.join(args.datasets)}_seed_{args.seed}"
    
    wandb_run = wandb.init(project="DAMIT_NEW[SimCLR]", config=cfg, name=FILENAME, dir=os.path.join(args.output_dir, FILENAME),
                           tags=args.datasets+['SimCLR'], group=FILENAME)
    
    train_dataset = make_monai_dataset_for_simclr(args.datasets, cfg)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        collate_fn=contrastive_collate_fn,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    cfg['model'].pop('n_classes', None)  # Remove n_classes if it exists, as it's not needed for SimCLR
    model = ViT3DSimCLR(out_dim=cfg['simclr']['out_dim'], **cfg['model'])
    
    optimizer = torch.optim.Adam(model.parameters(), cfg['optimizer']['lr'], weight_decay=cfg['optimizer']['weight_decay'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    with torch.cuda.device(int(args.devices)):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args, cfg=cfg, wandb_run=wandb_run, FILENAME=FILENAME)
        simclr.train(train_loader)
    
    wandb.finish()

if __name__ == "__main__":
    main()
