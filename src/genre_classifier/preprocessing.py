import os
import librosa
import json
from pathlib import Path
from math import ceil


def save_mfcc(
    dataset_path: str | Path,
    json_path: str | Path,
    samples_per_track: int,
    n_mfcc=13,
    n_fft=2048,
    hop_length=512,
    num_segments=5,
):
    """Creates a JSON file of the MFCCs for the dataset
    :param
    ----------
     dataset_path: path to the dataset folder
     json_path: name of JSON file to be created
     n_mfcc: number of MFCC coefficients to create
     n_fft:
     hop_length:
     num_segments:

    """
    # Create a dictionary to map semantic labels to numerical labels
    semantic_to_numeric = {}
    # dictionary to store data
    data = {
        "mappings": {},
        "mfcc": [],
        "labels": [],
    }
    num_samples_per_segment = int(samples_per_track / num_segments)
    expected_num_mfcc_vectors_per_segment = ceil(
        num_samples_per_segment / hop_length
    )  # round up always

    # Loop through all the data
    for i, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)):
        # dirpath = current folder path
        # dirnames = subfolders in dirpath
        # filenames = all files in dirpath

        # ensure that we're not at the root level (Audio folder)
        if dirpath != str(dataset_path):
            # save the semantic label
            dirpath_components = dirpath.split(os.sep)
            semantic_label = dirpath_components[-1]
            # Subtract 1 to skip the root folder
            semantic_to_numeric[i - 1] = semantic_label
            print(f"\nProcessing {semantic_label}")

            # process files
            for filename in filenames:
                # load audio file
                file_path = Path(dirpath, filename)  # os.path.join(dirpath, filename)
                try:
                    signal, sr = librosa.load(
                        file_path  # , sr=SAMPLE_RATE, duration=DURATION
                    )
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(
                        y=signal[start_sample:finish_sample],
                        sr=sr,
                        n_fft=n_fft,
                        n_mfcc=n_mfcc,
                        hop_length=hop_length,
                    )
                    mfcc = mfcc.T

                    # store mfcc for segment if it has expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        # can't save numpy arrays as json files
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print(f"{file_path}, segment:{s+1}")
    data["mappings"] = semantic_to_numeric
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
