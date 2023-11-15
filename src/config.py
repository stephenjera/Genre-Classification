from dataclasses import dataclass


@dataclass
class Hyperparameters:
    num_classes: int
    input_size: int
    hidden_size: int
    num_layers: int
    batch_size: int
    num_epochs: int
    learning_rate: float


@dataclass
class Params:
    model_name: str
    num_workers: int
    validation_size: float
    test_size: float


@dataclass
class Paths:
    dataset_path: str
    artifact_path: str
    conda_path: str
    code_path: str


@dataclass
class GenreClassifierConfig:
    hyperparameters: Hyperparameters
    params: Params
    paths: Paths
