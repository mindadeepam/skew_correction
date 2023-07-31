

import numpy as np
import timm
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

verbose=False

#######################################################################################################################################

class TimmClassifier(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=4, in_chans=1, dropout=0.0):
        super(TimmClassifier, self).__init__()
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes, 
            in_chans=in_chans,
            drop_rate=dropout
        )
        # self.lr = lr
        
    def forward(self, x):
        if verbose==True: print(x.shape)
        # out = F.relu(self.backbone(x))

        # out = F.softmax(out, dim=1)
        out = self.backbone(x)

        return out
    
    def predict(self, x):
        self.eval()  # Set the model to evaluation mode

        # we need 4 dims
        if len(x.shape)==3:
            x = x.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            _, predicted_classes = torch.max(probabilities, dim=1)
        
        if not predicted_classes.device==torch.device('cpu'):
            predicted_classes.to('cpu')
            
        return predicted_classes
    
    def load(self, ckpt_path, device='auto'):
        
        if device=='auto':
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_state_dict(torch.load(ckpt_path, map_location=device))


#######################################################################################################################################

class MyModelModule(pl.LightningModule):
    def __init__(self, model, loss_fn, lr):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # print(f"input shape {x.size()}, output shape {y.size()}")
        y_hat = self.model(x)
        # print(f"output {[np.argmax(_).item() for _ in y_hat.detach().cpu()]}, label {y}")
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        acc = get_acc(y_hat, y)
        self.log('train_acc', acc, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        acc = get_acc(y_hat, y)
        self.log('val_acc', acc, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).reshape(-1)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        print(f'==========> Epoch {self.current_epoch}')
        print_metrics_on_epoch_end(metrics)

#######################################################################################################################################

## utilities

def total_params(model):
    """gets total number of parameters in model"""
    return sum([param.numel() for param in model.parameters()])

def get_acc(y_hat, y):
    with torch.no_grad():
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).item() / y.size(0)
    return acc
        

def print_metrics_on_epoch_end(metrics):
    keys_to_print = ['train_loss', 'train_acc_epoch', 'val_loss', 'val_acc_epoch']
    try:
            
        filtered_dict = {key: round(metrics[key].item(),2) for key in keys_to_print}
        pretty_json = json.dumps(filtered_dict, indent=4)
        print(pretty_json)
    except:
        print({key: round(metrics[key].item(),2) for key in metrics})