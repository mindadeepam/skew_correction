

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

verbose=False

#######################################################################################################################################

class TimmClassifier(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=4, in_chans=1):
        super(TimmClassifier, self).__init__()
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes, 
            in_chans=in_chans
        )
        # self.lr = lr
        
    def forward(self, x):
        if verbose==True: print(x.shape)
        # out = F.relu(self.backbone(x))

        # out = F.softmax(out, dim=1)
        out = self.backbone(x)

        return out


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
        print(f"output {[np.argmax(_).item() for _ in y_hat.detach().cpu()]}, label {y}")
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
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


#######################################################################################################################################

## utilities

def total_params(model):
    """gets total number of parameters in model"""
    return sum([param.numel() for param in model.parameters()])