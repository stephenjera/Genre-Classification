# Create the model
import pytorch_lightning as pl
import yaml
import optuna
from genre_classifier.model import LSTMGenreModel, MFCCDataModule
from pathlib import Path


# Hyperparameters
NUM_CLASSES = 10  # number of genres
INPUT_SIZE = 13  # number of MFCC coefficients
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 64
NUM_EPOCHS = 55
LEARNING_RATE = 1e-3

NUM_WORKERS = 1
VALIDATION_SIZE = 0.25
TEST_SIZE = 0.2
DATASET_PATH = Path.cwd().parent / "data" / "processed" / "genres_mfccs.json"

ARTIFACT_PATH = "genre_classifier"
MODEL_NAME = "genre-classifier"

CONDA_PATH = Path.cwd().parent / "environment.yaml"
CODE_PATH = Path.cwd() / "genre_classifier"


def objective(trial):
    # Define the hyperparameters to be optimized
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    num_epochs = trial.suggest_int("num_epochs", 10, 100)

    # Create the model
    model = LSTMGenreModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        learning_rate=learning_rate,
        dataset_path=DATASET_PATH,
    )

    # Create the data loader
    dm = MFCCDataModule(
        dataset_path=DATASET_PATH,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        validation_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
    )

    # Train the model
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        log_every_n_steps=25,
        callbacks=[
            optuna.integration.PyTorchLightningPruningCallback(
                trial, monitor="val_accuracy"
            ),
        ],
    )
    trainer.fit(model, dm)

    # Evaluate the model and report the validation loss to Optuna
    output = trainer.validate(model, dm)
    val_accuracy = output[0]["val_accuracy"]
    trial.report(val_accuracy, step=trainer.current_epoch)
    return val_accuracy


if __name__ == "__main__":
    # Create a study object to manage the optimization
    study = optuna.create_study(direction="maximize")

    # Optimize the objective function
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    # Load original config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
        # Update hyperparameters
        config["hyperparameters"]["learning_rate"] = best_params["learning_rate"]
        config["hyperparameters"]["batch_size"] = best_params["batch_size"]
        config["hyperparameters"]["num_epochs"] = best_params["num_epochs"]

    # Write updated config to new file
    with open("config/config_best.yaml", "w") as f:
        yaml.dump(config, f)
