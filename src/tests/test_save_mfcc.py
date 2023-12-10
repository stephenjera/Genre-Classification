import json
import os
import shutil
from pathlib import Path

import pytest

from genre_classifier.preprocessing import save_mfcc


@pytest.fixture
def tmp_path():
    # Create the blues directory
    blues_dir = Path.cwd() / "subsection" / "genres" / "blues"
    Path.mkdir(blues_dir, parents=True, exist_ok=True)

    # Get the path to the genre directory
    genre_dir = Path.cwd().parent / "data" / "genres" / "blues"
    file_path = next(genre_dir.glob("**/*.wav"))

    dataset_path = Path.cwd() / "subsection" / "genres"

    # Copy the file to the subsection directory
    shutil.copy2(file_path, blues_dir)

    yield dataset_path

    # Teardown code
    if os.path.exists("data.json"):
        os.remove("data.json")
        print("Teardown happened: data.json file removed.")

    # Remove the 'subsection' directory and all its contents
    shutil.rmtree("subsection")
    print("Teardown happened: 'subsection' directory removed.")


@pytest.mark.usefixtures("tmp_path")
def test_save_mfcc_correct_num_coeffs(tmp_path):
    """Tests that the save_mfcc function saves the correct number of MFCC coefficients."""

    # Save the MFCCs
    save_mfcc(tmp_path, "data.json", 22050 * 30)

    # Load the MFCCs
    with open("data.json", "r") as fp:
        data = json.load(fp)

    # Assert that the number of MFCC coefficients is correct
    assert (
        len(data["mfcc"][0][0]) == 13
    )  # 13 is the default number of MFCC coefficients
