import os
import json
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class LSTMGenreModel(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        batch_size,
        learning_rate,
        dataset_path: str | Path,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate

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

    def prepare_data(self):
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
        ) = self.prepare_datasets(self.X, self.y, 0.25, 0.2)

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

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)
        # out shape: (batch, seq_len, hidden_size)

        # Take the final output and classify
        out = self.fc(out[:, -1, :])
        # out shape: (batch, num_classes)
        return out

    def training_step(self, batch):
        mfccs, labels = batch
        outputs = self(mfccs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        mfccs, labels = batch
        outputs = self(mfccs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        print(loss)
        self.log("val_loss", loss)

    # Test step
    def test_step(self, batch):
        X, y = batch

        # Forward pass
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)

        y_pred = torch.argmax(y_hat, dim=1)
        accuracy = (y_pred == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

    def save_checkpoint(self, checkpoint_path, filename="checkpoint.ckpt"):
        torch.save(self.state_dict(), os.path.join(checkpoint_path, filename))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
