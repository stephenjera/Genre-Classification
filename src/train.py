# Create the model
import pytorch_lightning as pl
import mlflow
import dagshub
import torch
from genre_classifier.model import LSTMGenreModel, MFCCDataModule
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from mlflow.models.signature import infer_signature

# Hyperparameters
NUM_CLASSES = 10  # number of genres
INPUT_SIZE = 13  # number of MFCC coefficients
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3

NUM_WORKERS = 1
VALIDATION_SIZE = 0.25
TEST_SIZE = 0.2
DATASET_PATH = Path.cwd().parent / "data" / "processed" / "genres_mfccs.json"

ARTIFACT_PATH = "genre_classifier"
MODEL_NAME = "genre-classifier"

CONDA_PATH = Path.cwd().parent / "environment.yaml"
CODE_PATH = Path.cwd() / "genre_classifier"

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
    # dagshub.init(
    #     repo_owner="stephenjera",
    #     repo_name="Genre-Classification",
    #     mlflow=True,
    # )
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.pytorch.autolog(log_models=False)

    # logger = TensorBoardLogger("tb_runs")
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_runs/profiler0"),
        schedule=torch.profiler.schedule(
            skip_first=10,
            wait=1,
            warmup=1,
            active=20,
        ),
    )
    trainer = pl.Trainer(
        # profiler=profiler,
        # logger=logger,
        max_epochs=NUM_EPOCHS,
        log_every_n_steps=25,
    )
    with mlflow.start_run():
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)

        input_tensor = torch.rand(1, 259, 13)
        predictions = model(input_tensor)
        input_np = input_tensor.numpy()
        predictions_np = predictions.detach().numpy()
        signature = infer_signature(input_np, predictions_np)

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=MODEL_NAME,
            conda_env=str(CONDA_PATH),
            code_paths=[str(CODE_PATH)],
            signature=signature,
            registered_model_name=MODEL_NAME,
            await_registration_for=0,
        )
