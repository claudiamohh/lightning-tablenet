import albumentations as A
import torch
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from datasets.dataset import Lightning_MarmotDataset
from models.model import Lightning_TableNet


image_size = (896, 896)

train_transform = A.Compose(
    [
        A.Resize(1024, 1024, always_apply=True),
        A.RandomResizedCrop(*image_size, scale=(0.7, 1.0), ratio=(0.7, 1)),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Normalize(),
        ToTensorV2(),
    ]
    )

test_transform = A.Compose(
    [A.Resize(*image_size, always_apply=True), A.Normalize(), ToTensorV2()]
)

dataset = Lightning_MarmotDataset(data_dir="./data/Marmot_data/", train_transform=train_transform, test_transform=test_transform, batch_size=2)

model = Lightning_TableNet(num_class=1, encoder='vgg')

checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, save_last=True, mode='min')      

early_stop_callback = EarlyStopping(monitor='val_loss', mode='min', patience=10)  

lr_monitor = LearningRateMonitor(logging_interval='step')   

trainer = pl.Trainer(accumulate_grad_batches=2,
                    gradient_clip_val=0.5, 
                    callbacks=[lr_monitor, checkpoint_callback, early_stop_callback], 
                    logger=TensorBoardLogger('lightning_tablenet', name="tablenet_baseline_adam_gradclipping"), 
                    max_epochs=5000,
                    gpus=1 if torch.cuda.is_available() else None)  

trainer.fit(model, 
            datamodule=dataset)

trainer.test(model,
             datamodule=dataset)
