# Adapted from https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torchmetrics import Accuracy, AUROC

# originally in config file
num_classes = 2
resnet_version = 18
learning_rate = 0.0001
batch_size = 32
transfer_learn = True
embedding_size = 512


class Model(pl.LightningModule):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters()

        self.lr = learning_rate
        self.batch_size = batch_size

        self.optimizer = Adam
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = Accuracy(
            task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes
        )
        self.auroc = AUROC(task="multiclass", num_classes=num_classes)

        # Using a pretrained ResNet backbone
        backbone = self.resnets[resnet_version](
            weights="DEFAULT" if transfer_learn else None
        )
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, X):
        x = self.feature_extractor(X)
        # Flatten the tensor for linear layer
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def _step(self, batch):
        x, y = batch["image"], batch["label"]
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)

        # Labels from logits - softmax prior?
        preds = torch.argmax(logits, dim=1)
        acc = self.acc(preds, y)

        # change this
        roc = self.auroc(logits, y)
        return loss, acc, roc

    def predict_step(self, batch, batch_idx):
        output = {}
        x, output["label"] = batch["image"], batch["label"]

        # essentially forward(), but we want to extract embeddings for MMD later
        x = self.feature_extractor(x)
        embeddings = x.view(x.size(0), -1)
        logits = self.classifier(embeddings)

        softmax = nn.Softmax(dim=1)
        return softmax(logits)

    def test_step(self, batch, batch_idx):
        loss, acc, roc = self._step(batch)
        # perform logging
        self.log(
            "test_loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_acc",
            acc,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_roc-auc",
            roc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        return roc
