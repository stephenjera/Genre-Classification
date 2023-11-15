import os
import json
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split


# Dataset
class MFCCDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        mfccs = self.X[idx]
        label = self.y[idx]
        return mfccs, label


class MFCCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path,
        batch_size,
        num_workers,
        test_size,
        validation_size,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.validation_size = validation_size

    @staticmethod
    def load_data(dataset_path):
        """
        Loads training dataset from json file.
            :param data_path (str): Path to json file containing data
            :return X (ndarray): Inputs
            :return y (ndarray): Targets
        """
        with open(dataset_path, "r") as fp:
            print("Loading Data")
            data = json.load(fp)
            # convert lists to numpy arrays
            X = np.array(data["mfcc"])
            # X = np.array(data["spectrogram"])
            y = np.array(data["labels"])
            mappings = data["mappings"]
            return X, y, mappings

    @staticmethod
    def prepare_datasets(
        X, y, test_size, validation_size, shuffle=True, random_state=42
    ):
        # create train, validation and test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
        )

        # create train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size
        )

        return X_train, y_train, X_test, y_test, X_val, y_val

    def setup(self, stage=None):
        # Load data
        self.X, self.y, _ = self.load_data(self.dataset_path)

        # Convert to tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

        # Create train/val/test split
        (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        ) = self.prepare_datasets(
            self.X,
            self.y,
            self.test_size,
            self.validation_size,
        )

        # Create dataset objects
        self.train_dataset = MFCCDataset(self.X_train, self.y_train)
        self.test_dataset = MFCCDataset(self.X_test, self.y_test)
        self.val_dataset = MFCCDataset(self.X_val, self.y_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


class LSTMGenreModel(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        learning_rate: float,
        dataset_path: str | Path,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.dataset_path = dataset_path
        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

    def _common_step(self, batch):
        mfccs, labels = batch
        outputs = self(mfccs)
        loss = self.loss_fn(outputs, labels)
        return loss, outputs, labels

    def training_step(self, batch):
        loss, outputs, labels = self._common_step(batch)
        accuracy = self.accuracy(outputs, labels)
        f1_score = self.f1_score(outputs, labels)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "outputs": outputs, "labels": labels}

    def validation_step(self, batch):
        loss, outputs, labels = self._common_step(batch)
        self.log("val_loss", loss)
        return loss

    # Test step
    def test_step(self, batch):
        loss, outputs, labels = self._common_step(batch)
        self.log("test_loss", loss)
        return loss

    def save_checkpoint(self, checkpoint_path, filename="checkpoint.ckpt"):
        torch.save(self.state_dict(), os.path.join(checkpoint_path, filename))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
