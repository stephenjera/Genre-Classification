import os
import pytest
import numpy.testing as npt
import json
from genre_classifier.model import MFCCDataModule


def test_load_data_valid_json():
    """
    Test loading data from a valid JSON file.
    """

    # Create sample JSON data
    json_data = {
        "mfcc": [[1, 2, 3], [4, 5, 6]],
        "labels": [0, 1],
        "mappings": [
            {"label": 0, "name": "cat"},
            {"label": 1, "name": "dog"},
        ],
    }

    # Save JSON data to a temporary file
    with open("tmp.json", "w") as fp:
        json.dump(json_data, fp)

    # Load data from the temporary file
    X, y, mappings = MFCCDataModule.load_data("tmp.json")

    # Check if the loaded data is correct
    npt.assert_array_equal(X, [[1, 2, 3], [4, 5, 6]])
    npt.assert_array_equal(y, [0, 1])
    assert mappings == [
        {"label": 0, "name": "cat"},
        {"label": 1, "name": "dog"},
    ]

    # Delete the temporary file
    os.remove("tmp.json")


@pytest.mark.parametrize("missing_key", ["mfcc", "labels", "mappings"])
def test_load_data_missing_required_key(missing_key):
    """Test loading data with missing key."""

    # Create JSON data with missing key
    data = {
        "mfcc": [[1, 2, 3], [4, 5, 6]],
        "labels": [0, 1],
        "mappings": [{"label": 0, "name": "cat"}, {"label": 1, "name": "dog"}],
    }
    del data[missing_key]

    # Save to temporary file
    with open("tmp.json", "w") as f:
        json.dump(data, f)

    # Attempt to load data and assert error
    with pytest.raises(KeyError):
        MFCCDataModule.load_data("tmp.json")

    # Clean up
    os.remove("tmp.json")

def test_load_data_extra_keys():
    # Create JSON data with missing key
    data = {
        "mfcc": [[1, 2, 3], [4, 5, 6]],
        "test": [1,1],
        "labels": [0, 1],
        "mappings": [{"label": 0, "name": "cat"}, {"label": 1, "name": "dog"}],
    }
    # Save to temporary file
    with open("tmp.json", "w") as f:
        json.dump(data, f)

    X, y, mappings = MFCCDataModule.load_data("tmp.json")

    # Assert extra key was ignored
    assert "extra_key" not in X
    assert "extra_key" not in y
    assert "extra_key" not in mappings

    # Clean up
    os.remove("tmp.json")