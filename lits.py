from loguru import logger

import torch

import lightning as L
from torchmetrics.classification import Accuracy, F1Score, Recall

class LitMAE(L.LightningModule):
    def __init__(self, 
                 model,
                 mask_ratio,
                 **kwargs):
        super().__init__()
        self.model = model
        self.mask_ratio = mask_ratio
        
        # optimizer parameters
        self.epochs = kwargs.get('epochs', 50)
        self.hparams = kwargs.get('hparams', 
                                  {'lr': 0.0001,
                                   'weight_decay': 0.0001,
                                   'warmup_epochs': 40,
                                   'min_lr': 0.0000001,
                                   'betas': (0.9, 0.95)})
        
        self.save_hyperparameters()
        
    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        x = batch['image']
        loss, _, _ = self(x, mask_ratio=self.mask_ratio)  # assuming mask_ratio is fixed for training
        self.log_dict({'loss': loss,
                       'lr': self.optimizers().param_groups[0]['lr']},
                        prog_bar=True)
        return loss
                
    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), 
                                      lr=self.hparams['SOLVER']['lr'], 
                                      weight_decay=self.hparams['SOLVER']['weight_decay'],
                                      betas=self.hparams['SOLVER']['betas'])
             
    def on_train_epoch_start(self):
        # Adjust learning rate before the training epoch starts
        self._adjust_learning_rate_halfcosine()
        
    def _adjust_learning_rate_halfcosine(self):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if self.current_epoch < self.hparams['SOLVER']['warmup_epochs']:
            lr = self.hparams['SOLVER']['lr'] * self.current_epoch / self.hparams['SOLVER']['warmup_epochs'] 
        else:
            lr = self.hparams['SOLVER']['min_lr'] + \
                (self.hparams['SOLVER']['lr'] - self.hparams['SOLVER']['min_lr']) * 0.5 * \
                (1. + torch.cos(torch.pi * (self.current_epoch - self.hparams['SOLVER']['warmup_epochs']) / \
                (self.epochs - self.hparams['SOLVER']['warmup_epochs'])))
        for param_group in self.optimizers().param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
    
class LitViT(L.LightningModule):
    def __init__(self, 
                 model,
                 num_classes,
                 loss_fn,
                 learning_rate,
                 mode,
                 **kwargs):
        super().__init__()
        self.model = model
        self.num_classes = num_classes if num_classes is not None else 2
        if kwargs.get('pretrained_model_path') is not None:
            self.copy_pretrained_weights(kwargs['pretrained_model_path'])
        # self.copy_pretrained_weights() ##
        
        # optimizer parameters
        self.learning_rate = learning_rate
        self.weight_decay = kwargs.get('weight_decay', 0.0001)
        self.betas = kwargs.get('betas', (0.9, 0.999))
        self.epochs = kwargs.get('epochs', 50)
        
        self.mode = mode
        
        self.save_hyperparameters()
        # loss
        self.loss_fn = loss_fn
        self.train_acc_fn = Accuracy(task='multiclass', num_classes=self.num_classes, average='micro')
        self.val_acc_fn = Accuracy(task='multiclass', num_classes=self.num_classes, average='micro') # same as binary
        self.f1_fn = F1Score(task='multiclass', num_classes=self.num_classes, average=None) # same as binary
        self.recall_fn = Recall(task='multiclass', num_classes=self.num_classes, average=None) # same as binary
        
        # logs
        self.correct = []
        self.val_accs, self.recalls, self.f1s, self.corrects = [], [], [], []
        
    def copy_pretrained_weights(self, pre_trained_model_path):
        # load pretrained weights
        checkpoint = torch.load(pre_trained_model_path, map_location='cpu')
        if pre_trained_model_path.endswith('.pth.tar'):
            checkpoint_model = checkpoint['net']
        elif pre_trained_model_path.endswith('.ckpt'):
            checkpoint_model = checkpoint['state_dict']
        # remove keys that are not in the model
        keys_to_remove = ['head.weight', 'head.bias', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']
        state_dict = self.model.state_dict()
        for key in keys_to_remove:
            if key in checkpoint_model and key in state_dict and state_dict[key].shape != checkpoint_model[key].shape:
                print(f"Removing key {key} from pretrained checkpoint")
                del checkpoint_model[key]
        # load the model
        msg = self.model.load_state_dict(checkpoint_model, strict=False)
        logger.critical(f'Loaded pretrained weights from {pre_trained_model_path}')
        logger.critical(f'Keys not loaded (missing keys): {msg.missing_keys}')            
    
    def on_train_epoch_end(self):
        # get max hard dice score across steps
        train_acc = self.train_acc_fn.compute().item()
        self.log('train_acc', train_acc, prog_bar=True)
        self.train_acc_fn.reset()
        
    def on_validation_epoch_end(self):
        # get max hard dice score from validation
        val_acc = self.val_acc_fn.compute().item()
        recall = self.recall_fn.compute()[1].item() # [1] for class 1
        f1 = self.f1_fn.compute()[1].item() # [1] for class 1
        
        self.log_dict({'val_acc': val_acc,
                        'recall': recall,
                        'f1': f1}, prog_bar=True)
        # log metrics to array
        self.val_accs.append(val_acc)
        self.recalls.append(recall)
        self.f1s.append(f1)
        self.corrects.append(sum(self.correct))  # sum of corrects across all batches for this epoch
        
        # we have to reset the metrics after logging
        self.val_acc_fn.reset()
        self.recall_fn.reset()
        self.f1_fn.reset()
        self.correct = []  # reset correct for the next epoch
    
    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        logits = self(x)
        y_hat = torch.argmax(logits, dim=1)
        # compute loss
        loss = self.loss_fn(logits, y)
        # accumulate accuracy
        self.train_acc_fn.update(y_hat, y)
        self.log_dict({'loss': loss,
                       'lr': self.optimizers().param_groups[0]['lr']}, 
                      prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        logits = self(x)
        y_hat = torch.argmax(logits, dim=1)
        # compute loss
        loss = self.loss_fn(logits, y)
        # accumulate accuracy, f1, recall
        self.val_acc_fn.update(y_hat, y)
        self.recall_fn.update(y_hat, y)
        self.f1_fn.update(y_hat, y)
        self.correct.append(y_hat.eq(y).sum().item())
        self.log('val_loss', loss, prog_bar=True)
                
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), 
                                      lr=self.learning_rate, 
                                      weight_decay=self.weight_decay,
                                      betas=self.betas)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs),
                # 'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=70, eta_min=1e-6),
                'interval': 'epoch',
                'frequency': 1
            }
        }