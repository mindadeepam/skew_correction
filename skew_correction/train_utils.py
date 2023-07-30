import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
from pytz import timezone  # This library helps you get the timezone if needed
import os

from skew_correction.constants import root_dir

# configure loggers and callbacks
current_date = datetime.now(timezone('Asia/Kolkata')).strftime('%Y-%m-%d')



checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(root_dir, 'checkpoints'),
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    filename="{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}"
)
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=0.00, patience=4, verbose=False, mode="min"
)