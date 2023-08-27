import numpy as np
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from skimage.io import imread
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Dict


class Dataset(VisionDataset):
    def __init__(self, root: str, dataframe: pd.DataFrame, transform: Callable) -> None:
        """
        Torchvision dataset for loading images and metadata.

        Args:
            root (str): Directory where images are stored
            dataframe (pd.DataFrame): Dataframe mapping image filenames to labels
            transform (Callable): Transform to apply to image
        """
        super().__init__(root=root, transform=transform)
        self.root = Path(self.root)
        self.dataset_dataframe = dataframe
        self.labels = self.dataset_dataframe.label.values
        self.subject_ids = self.dataset_dataframe.image.values
        self.filenames = [self.root / subject_id for subject_id in self.subject_ids]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Iterator to get images and their labels from the dataset.

        Args:
            index (int): Index of image from list of filenames

        Returns:
            Dict[str, Any]: Dictionary with image and label
        """
        filename = self.filenames[index]
        image = imread(filename).astype(np.uint8)
        # Added to convert from 1 channel to 3 channels of grayscale
        image = np.repeat(image[..., np.newaxis], 3, -1)

        return {
            "image": self.transform(image),
            "label": self.labels[index],
        }

    def __len__(self) -> int:
        """
        Finds the total size of the dataset.

        Returns:
            int: Size of dataset
        """
        return len(self.filenames)


class DataModule(pl.LightningDataModule):
    def __init__(
        self, images_dir: str, metadata_path: str, batch_size: int, num_workers: int = 8
    ) -> None:
        """
        Data module to load images and metadata.

        Args:
            images_dir (str): Directory where images are stored
            metadata_path (str): Path where metadata (image filenames, labels) are stored
            batch_size (int): Batch size for inference
            num_workers (int): Number of workers to use
        """
        super().__init__()
        self.image_dir = images_dir
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.df = pd.read_csv(self.metadata_path)

    def predict_dataloader(self, transform: Callable) -> DataLoader:
        """
        Dataloader for prediction.

        Args:
            transform (Callable): Transform to apply to image

        Returns:
            Dataloader: Dataloader for dataset with transform applied
        """
        self.dataset_predict = Dataset(
            self.image_dir, dataframe=self.df, transform=transform
        )
        return DataLoader(
            self.dataset_predict,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
