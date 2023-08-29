import pytorch_lightning as pl
import torch
from typing import Callable


class Model(pl.LightningModule):
    def __init__(self) -> None:
        """
        Setup and initialisation for Model class
        """
        pass

    def configure_optimizers(self) -> Callable:
        """
        Configures the optimiser.
        """
        pass

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Step for prediction/inference, returning the softmax for shift detection
        """
        pass

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """
        Step for testing, calculating loss, accuracy and ROC-AUC.
        """
        pass
