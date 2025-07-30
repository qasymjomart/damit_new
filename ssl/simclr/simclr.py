import os
from loguru import logger

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from utils import accuracy, save_checkpoint

class SimCLR(object):

    def __init__(self, **kwargs):
        self.FILENAME = kwargs['FILENAME']
        self.rank = kwargs['rank']
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.cfg = kwargs['cfg']
        self.wandb_run = kwargs['wandb_run']
        logger.add(os.path.join(self.args.output_dir, self.FILENAME, 'simclr.log'), level='DEBUG')
        self.criterion = torch.nn.CrossEntropyLoss().to(self.rank)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.cfg['training']['batch_size']) for i in range(self.cfg['simclr']['n_views'])], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.rank)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.rank)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.rank)

        logits = logits / self.cfg['simclr']['temperature']
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler('cuda', enabled=self.cfg['training']['use_fp16'])

        logger.info(f"Configs: \n {self.cfg}")
        n_iter = 0
        logger.info(f"Start SimCLR training for {self.cfg['training']['epochs']} epochs.")

        for epoch_counter in range(self.cfg['training']['epochs']):
            train_loader.sampler.set_epoch(epoch_counter)
            for images1, images2 in tqdm(train_loader):
                images = torch.cat((images1, images2), dim=0)
                images = images.to(self.rank)

                with autocast('cuda', enabled=self.cfg['training']['use_fp16']):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.cfg['simclr']['log_every_n_steps'] == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.wandb_run.log({'loss': loss.item()}, step=n_iter)
                    self.wandb_run.log({'acc/top1': top1[0].item()}, step=n_iter)
                    self.wandb_run.log({'acc/top5': top5[0].item()}, step=n_iter)
                    self.wandb_run.log({'learning_rate': self.scheduler.get_last_lr()[0]}, step=n_iter)
                    logger.info(f"Iter: {n_iter}\tLoss: {loss.item()}\tTop1 accuracy: {top1[0].item()}\tTop5 accuracy: {top5[0].item()}")

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logger.success(f"Epoch: {epoch_counter}\tLoss: {loss.item()}\tTop1 accuracy: {top1[0].item()}")

        logger.success("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.cfg['training']['epochs'])
        if self.rank == 0:
            torch.save(self.model.module.state_dict(), os.path.join(self.args.output_dir, self.FILENAME, checkpoint_name))
            # save_checkpoint({
                # 'epoch': self.cfg['training']['epochs'],
                # 'state_dict': self.model.module.state_dict(),
                # 'optimizer': self.optimizer.state_dict(),
            # }, is_best=False, filename=os.path.join(self.args.output_dir, self.FILENAME, checkpoint_name))
        logger.success(f"Model checkpoint and metadata has sbeen saved at {self.args.output_dir}{self.FILENAME}.")
