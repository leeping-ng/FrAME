import numpy as np
import pandas as pd
import pytorch_lightning as pl

from pathlib import Path
from skimage.io import imread
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import VisionDataset


class Dataset(VisionDataset):
    def __init__(self, root, dataframe, transform):
        """
        Torchvision dataset for loading RSNA dataset.
        Args:
            root: the data directory where the images can be found
            dataframe: the csv file mapping patient id, metadata, file names and label.
            transform: the transformation (i.e. preprocessing and / or augmentation) to apply to the image after loading them.

        This dataset returns a dictionary with the image data, label and metadata.
        """
        super().__init__(root=root, transform=transform)
        self.root = Path(self.root)
        self.dataset_dataframe = dataframe
        self.labels = self.dataset_dataframe.label.values.astype(np.int64)
        self.subject_ids = self.dataset_dataframe.image.values
        self.filenames = [self.root / subject_id for subject_id in self.subject_ids]

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        image = imread(filename).astype(np.uint8)
        # Added to convert from 1 channel to 3 channels of grayscale
        image = np.repeat(image[..., np.newaxis], 3, -1)

        return {
            "image": self.transform(image),
            "label": self.labels[index],
        }

    def __len__(self) -> int:
        return len(self.filenames)


class DataModule(pl.LightningDataModule):
    def __init__(self, images_dir, metadata_path, batch_size, num_workers=8):
        super().__init__()
        self.image_dir = images_dir
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.df = pd.read_csv(self.metadata_path)

    def predict_dataloader(self, transform):
        self.dataset_predict = Dataset(
            str(self.image_dir),
            dataframe=self.df,
            transform=transform,
        )
        return DataLoader(
            self.dataset_predict,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
