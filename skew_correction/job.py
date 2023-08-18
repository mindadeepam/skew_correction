""" Testing script for training script using vertex ai,
this contains the latest code for modelling """

from skew_correction.data import DatasetClass, MyDataModule, plot_random_images
from skew_correction.model import MyModelModule, TimmClassifier, total_params
from skew_correction.constants import root_dir, device
from skew_correction.train_utils import checkpoint_callback, early_stop_callback

import os
import torch
import timm 
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
from pytz import timezone  # This library helps you get the timezone if needed


# # lines for local training 
# data_dir = '/home/jupyter/farmart/maize_quality_detection/data/maize/grains/new_grains'
# outputdir = '.'
# loggerdir = '.'


# # lines to change for a vertex-ai training job
# =============================================================================================
data_dir = "/gcs/fmt-ds-bucket/dev/grain_classification_model_data/grains/maize2/new_grains/"

outputdir = os.getenv('CHECKPOINT_DIR')
loggerdir = os.getenv('LOGGER_DIR') 


dataset = DatasetClass("/home/deepam_minda_farmart_co/fmt/skew_correction/data/train_data.csv", split='train')
plot_random_images(dataset)
pl_data = MyDataModule(dataset, train_bs=16, val_bs=8)

## check data sizes
pl_data.setup()
tdl = pl_data.train_dataloader()
vdl = pl_data.val_dataloader()
print(f"len of dataloader = {len(tdl)}, batchsize = {tdl.batch_size} \ntotal samples = {len(tdl)*tdl.batch_size}")
print(f"len of dataloader = {len(vdl)}, batchsize = {vdl.batch_size} \ntotal samples = {len(vdl)*vdl.batch_size}")


## load model

lr = 0.0001
loss_fn = torch.nn.CrossEntropyLoss()

model_string = 'mobilenetv3_large_100'
dropout=0.3
model = TimmClassifier(model_string, dropout=dropout)
print(f"total_params: {total_params(model)}")

pl_model = MyModelModule(model, loss_fn, lr)

verbose=False

# Get the current date and time in a specific format
current_date = datetime.now(timezone('Asia/Kolkata')).strftime('%Y-%m-%d')

tb_logger = TensorBoardLogger(
    save_dir=os.path.join(root_dir, 'logs'), 
    name=f"{current_date}-{model_string}-{round(total_params(model)/1000000,2)}m_params-{len(tdl)*tdl.batch_size}samples-lr{lr}-bs{tdl.batch_size}-cpu-drop{dropout}"
)
print(tb_logger.name)
trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        max_epochs=50, 
        logger=tb_logger, 
        log_every_n_steps=1, 
        limit_train_batches=None, 
        limit_val_batches=None,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

trainer.fit(pl_model, datamodule=pl_data)

# pl_model.trainer.callback_metrics
