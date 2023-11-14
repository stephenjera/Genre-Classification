import torch
from genre_classifier.model import MFCCDataset


def test_mfcc_dataset_init():
    # Create some sample data
    X = torch.randn(10, 20)
    y = torch.randint(0, 3, (10,))

    # Create the dataset
    dataset = MFCCDataset(X, y)

    # Check that the dataset has the correct number of samples
    assert len(dataset) == len(y)


def test_mfcc_dataset_getitem():
    # Create some sample data
    X = torch.randn(10, 20)
    y = torch.randint(0, 3, (10,))

    # Create the dataset
    dataset = MFCCDataset(X, y)

    # Get a sample item from the dataset
    mfccs, label = dataset[0]

    # Check that the MFCCs have the correct shape
    assert mfccs.shape[0] == 20

    # Check that the label is a scalar
    assert label.ndim == 0
