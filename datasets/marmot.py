from typing import List
from pathlib import Path

import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A
import pytorch_lightning as pl
from albumentations import Compose
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class MarmotDataset(Dataset):
    """Custom Dataset with Marmot Dataset."""

    def __init__(self, data=List[Path], transforms: Compose = None) -> None:
        """
        Args:
            data (List[Path]): A list of Path
            transforms (Optional[Compose]): Optional albumentations to be applied on a sample.
        """
        self.data = data
        self.transforms = transforms

    def __len__(self):
        """Returns the size of Marmot Dataset"""
        return len(self.data)

    def __getitem__(self, item):
        """
        Function to return a sample from the Marmot dataset given an index after undergoing albumentations.

        Args:
            item (int): unique filenames

        Returns (Tuple[tensor, tensor, tensor]): Image, Table Mask, Column Mask

        Usage example:
          └── data/
            ├── marmot_data/
            │   ├── 200_12.bmp
            │   ├── 200_13.bmp
            ├── table_mask/
            │   ├── 200_12.bmp
            │   └── 200_13.bmp
            └── column_mask/
                ├── 200_12.bmp
                └── 200_13.bmp

            dataset = MarmotDataset(data=list(Path('/data/Marmot_data/').rglob("*.bmp")))
            dataset[0]
            >> (array([tensor], array[tensor], array[tensor])
        """

        unique_filename = self.data[item].stem

        image_path = self.data[item]
        table_path = self.data[item].parent.parent.joinpath(
            "table_mask", unique_filename + ".bmp"
        )
        column_path = self.data[item].parent.parent.joinpath(
            "column_mask", unique_filename + ".bmp"
        )

        image = np.array(Image.open(image_path))
        table_mask = np.expand_dims(np.array(Image.open(table_path)), axis=2)
        column_mask = np.expand_dims(np.array(Image.open(column_path)), axis=2)

        mask = np.concatenate([table_mask, column_mask], axis=2) / 255
        sample = {
            "image": image,
            "mask": mask,
        }
        if self.transforms:
            sample = self.transforms(image=image, mask=mask)

        image = sample["image"]
        mask_table = sample["mask"][:, :, 0].unsqueeze(0)
        mask_column = sample["mask"][:, :, 1].unsqueeze(0)
        return image, mask_table, mask_column


class LightningMarmotDataset(pl.LightningDataModule):
    """To create a Lightning Marmot Dataset from Marmot Dataset"""

    def __init__(
        self,
        data_dir: str = "/data",
        train_transform: Compose = None,
        test_transform: Compose = None,
        batch_size: int = 8,
        num_workers: int = 2,
    ):
        """
        Formatting LightningMarmotDataset and allowing LightningMarmotDataset to initialize the attributes.
        The following attributes are: data, train_transform, test_transform, batch_size and num_workers.

        Args:
            data_dir (str): Dataset Directory
            train_transform (Optional[Compose]): Albumentations to be applied on training dataset.
            test_transform (Optional[Compose]): Alumentations to be applied on validation and testing dataset.
            batch_size (int): Define batch size, by default: 8
            num_worker(int): Define number of works to process data, by default: 2
        """

        super().__init__()
        self.data = list(Path(data_dir).rglob("*.bmp"))
        self.train_augmentation = train_augementation
        self.test_processing = test_processing
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.setup()

    def setup(self, stage: str = None) -> None:
        """Function to create training, validation and test sets by slicing dataset.

        Training set: first 80% of total sample
        Validation set: next 10% of total sample (80%-90%)
        Testing set: last 10% of the sample

        Args:
            stage(Optional[str]): Used to seperate setup logic for trainer.fit and trainer.test
        """

        n_samples = len(self.data)
        self.data.sort()
        train_size = slice(0, int(n_samples * 0.8))
        val_size = slice(int(n_samples * 0.8), int(n_samples * 0.9))
        test_size = slice(int(n_samples * 0.9), n_samples)

        self.dataset_train = MarmotDataset(
            self.data[train_size], transforms=self.train_augmentation
        )
        self.dataset_val = MarmotDataset(
            self.data[val_size], transforms=self.test_processing
        )
        self.dataset_test = MarmotDataset(
            self.data[test_size], transforms=self.test_processing
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
