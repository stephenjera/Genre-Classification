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

## Model 

A LSTM neural network implemented in PyTorch is used for sequence classification. The model architecture consists of:

- LSTM input layer
- Fully connected output layer
- CrossEntropyLoss objective
- Adam optimizer

The MFCC audio features are fed into the LSTM layer, and the output is a predicted genre label.

## Tensorboard
The project uses Tensorboard to visualise some graphs to see these in the root directory run after code executuion then naviagte to localhos:6006 in a browser
```
tensorboard --logdir=notebooks/runs
```

## DVC 
[DVC documentation](https://dvc.org/doc/start)

Install dvc and dvc-s3
```shell
pipenv install dvc dvc-s3 -d
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

To add a remote go to the repository in DagsHub and select the dvc option and follow the instructions.

Commit Changes
```shell
git add .  git commit -m "message"
```
```shell
git commit -m "message"
```

Push to DagsHub
```shell
git push origin master
```
Push Data to DVC Remote
```shell
dvc push -r origin
```

Pull Changes
```shell
git pull origin master
```
```shell
dvc pull -r origin
```

## Installing Pytorch 

To create a conda enviornment install Anaconda. Open a new anaconda terminal and navigate to the root directory and run the following code:

```shell
conda env create -f environment.yaml
```
```shell
conda activate genre-classification
```

To create the environment from scratch run the following:

```shell
conda create -n genre-classification python=3.10.11
```

```shell
conda activate genre-classification
```

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```


To save the conda env use 
```shell
conda env export > environment.yaml
```
and then remove the prefix section at the end

CUDA version 11.8 was used, a tutorial on how to set up the CUDA toolkit can be found [here](https://www.youtube.com/watch?v=r7Am-ZGMef8)