# Create the model
import pytorch_lightning as pl
import mlflow
import dagshub
import torch
from model import LSTMGenreModel, MFCCDataModule
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler


# Hyperparameters
NUM_CLASSES = 10  # number of genres
INPUT_SIZE = 13  # number of MFCC coefficients
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

NUM_WORKERS = 1
VALIDATION_SIZE = 0.25
TEST_SIZE = 0.2
DATASET_PATH = Path.cwd() / "src" / "data.json"

model = LSTMGenreModel(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    learning_rate=LEARNING_RATE,
    dataset_path=DATASET_PATH,
)

dm = MFCCDataModule(
    dataset_path=DATASET_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    validation_size=VALIDATION_SIZE,
    test_size=TEST_SIZE,
)

if __name__ == "__main__":
    # dagshub.init(repo_owner="stephenjera", repo_name="Genre-Classification", mlflow=True)
    # mlflow.pytorch.autolog()
    logger = TensorBoardLogger("tb_runs")
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_runs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    )
    trainer = pl.Trainer(
        profiler=profiler,
        logger=logger,
        max_epochs=NUM_EPOCHS,
        log_every_n_steps=25,
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
    trainer.save_checkpoint("checkpoint.ckpt")
