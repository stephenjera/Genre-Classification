# Music Genre Classification

This project implements a LSTM neural network for classifying music by genre based on audio features.

## Data

The [GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/) dataset is used, which contains 1000 audio tracks evenly split across 10 genres:

- Blues
- Classical  
- Country
- Disco
- Hiphop
- Jazz  
- Metal
- Pop
- Reggae
- Rock

The raw .wav files are preprocessed to extract Mel-Frequency Cepstral Coefficients (MFCCs) which represent the short-term power spectrum of the audio.

## Contents

1. [Model](#model)
1. [Tensorboard](#tensorboard)
1. [DVC (Data Version Control)](#dvc)
1. [Installing PyTorch GPU](#installing-pytorch-gpu)
1. [MLflow](#mlflow)
1. [Pytest](#pytest)
1. [Dev Container](#dev-container)

## Model

A LSTM neural network implemented in PyTorch is used for sequence classification. The model architecture consists of:

- LSTM input layer
- Fully connected output layer
- CrossEntropyLoss objective
- Adam optimizer

The MFCC audio features are fed into the LSTM layer, and the output is a predicted genre label.

[Return to Top](#contents)

## Tensorboard

The project uses Tensorboard to visualise some graphs to see these in the root directory run after code executuion then naviagte to localhos:6006 in a browser

```shell
tensorboard --logdir=notebooks/runs
```

[Return to Top](#contents)

## DVC

[DVC documentation](https://dvc.org/doc/start)

Install dvc and dvc-s3

```shell
pip install dvc dvc-s3 -d
```

Initialize DVC in your local project directory

```shell
dvc init
```

This will create a .dvc directory that will store all the DVC related files.

Add Data to DVC

```shell
dvc add data/.
```

This will create a .dvc file that tracks your data.

To add a remote or setup credentials go to the repository in DagsHub and click the remote button, then select the dvc option and follow the instructions.

Commit Changes

```shell
git add .  git commit -m "message"
git commit -m "message"
```

Push to DagsHub

```shell
git push origin main
```

Push Data to DVC Remote

```shell
dvc push -r origin
```

Pull Changes

```shell
git pull origin main
dvc pull -r origin
```

[Return to Top](#contents)

## Installing Pytorch GPU

To create a conda enviornment install Anaconda. Open a new anaconda terminal and navigate to the root directory and run the following code:

```shell
conda env create -f environment.yaml
conda activate genre-classification
```

To create the environment from scratch run the following:

```shell
conda create -n genre-classification python=3.10.11
conda activate genre-classification
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install required packages

```shell
pip install <packages>
```

To save the conda environment run *export_conda_env.py* from the scripts directory

```shell
python export_conda_env.py
```

CUDA version 11.8 was used, you need both the drivers and toolkit.
Once the drivers are installed you need to run the following in the devcontainer.

- [Setup drivers on windows](https://www.youtube.com/watch?v=r7Am-ZGMef8)
- [Nvdia toolkit archvive](https://developer.nvidia.com/cuda-toolkit-archive)

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-debian11-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo dpkg -i cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda
export PATH=$PATH:/usr/local/cuda/bin
```

To test it run

```shell
nvidia-smi
nvcc --version
```

[Return to Top](#contents)

## MLflow

Create local testing server

```shell
mlflow server
```

Set environment variables

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/stephenjera/Genre-Classification.mlflow"
```

Serve model from remote

```shell
mlflow models serve -m "models:/genre-classifier/<version>" --port 1234 --no-conda
```

[Return to Top](#contents)

## Pytest

Run in src folder  

```shell
python -m pytest
```

[Return to Top](#contents)

## Dev Container

Ensure Docker is running

Run the dev container ```ctl + shift + p``` and search for **dev container** and click the option to open devcontainer

Open a new terminal

```shell
docker ps
docker exec -it <container_id>  /bin/zsh
```

[Return to Top](#contents)
