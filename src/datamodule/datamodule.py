from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from PIL import Image, ImageFile
import os

from .dataset import ImageDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageForgeryDatamMdule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path,
                 num_workers: int = 4,
                 train_batch_size: int = 32,
                 val_batch_size: int = 16):
        super().__init__()
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageDataset(
                dataset_path=self.dataset_path,
                file_name="train_df.pkl"
            )
            self.val_dataset = ImageDataset(
                dataset_path=self.dataset_path,
                file_name="val_df.pkl"
            )

        if stage == 'test' or stage is None:
            self.test_dataset = ImageDataset(
                dataset_path=self.dataset_path,
                file_name="test_df.pkl"
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.val_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.val_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          shuffle=False)
