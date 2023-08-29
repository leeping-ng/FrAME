# Adapted from https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torchmetrics import Accuracy, AUROC
from typing import Callable

# kept here separately from config file for modularity if model is to be swapped
NUM_CLASSES = 2
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
TRANSFER_LEARN = True


class Model(pl.LightningModule):
    def __init__(self) -> None:
        """
        Setup and initialisation for Model class
        """
        super().__init__()

        self.save_hyperparameters()
        self.optimizer = Adam
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = Accuracy(
            task="binary" if NUM_CLASSES == 2 else "multiclass", num_classes=NUM_CLASSES
        )
        self.auroc = AUROC(task="multiclass", num_classes=NUM_CLASSES)

        backbone = models.resnet18(weights="DEFAULT" if TRANSFER_LEARN else None)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, NUM_CLASSES)

    def configure_optimizers(self) -> Callable:
        """
        Configures the optimiser.

        Returns:
            optimiser (Callable): Optimiser object
        """
        return self.optimizer(self.parameters(), lr=LEARNING_RATE)

    def _step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward step to obtain logits.

        Args:
            x (torch.Tensor): Batched image data

        Returns:
            logits (torch.Tensor): Predicted logits
        """
        x = self.feature_extractor(x)
        # Flatten the tensor for linear layer
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Step for prediction/inference, returning the softmax for shift detection

        Args:
            batch (torch.Tensor): Batched data
            batch_idx (int): Index of item in batch

        Returns:
            softmax (torch.Tensor): Softmax output from prediction
        """
        x, _ = batch["image"], batch["label"]
        logits = self._step(x)
        softmax = nn.Softmax(dim=1)
        return softmax(logits)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """
        Step for testing, calculating loss, accuracy and ROC-AUC.

        Args:
            batch (torch.Tensor): Batched data
            batch_idx (int): Index of item in batch

        Returns:
            roc (float): ROC-AUC on test data
        """
        x, y = batch["image"], batch["label"]
        logits = self._step(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.acc(preds, y)
        roc = self.auroc(logits, y)

        self.log(
            "test_loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=BATCH_SIZE,
        )
        self.log(
            "test_acc",
            acc,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=BATCH_SIZE,
        )
        self.log(
            "test_roc-auc",
            roc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=BATCH_SIZE,
        )
        return roc
