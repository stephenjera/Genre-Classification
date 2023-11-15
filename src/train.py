# Create the model
import pytorch_lightning as pl
import mlflow
import dagshub
import torch
import hydra
from hydra.core.config_store import ConfigStore
from config import GenreClassifierConfig
from genre_classifier.model import LSTMGenreModel, MFCCDataModule
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from mlflow.models.signature import infer_signature


cs = ConfigStore.instance()
cs.store(name="genre_config", node=GenreClassifierConfig)


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: GenreClassifierConfig):
    model = LSTMGenreModel(
        input_size=cfg.hyperparameters.input_size,
        hidden_size=cfg.hyperparameters.hidden_size,
        num_layers=cfg.hyperparameters.num_layers,
        num_classes=cfg.hyperparameters.num_classes,
        learning_rate=cfg.hyperparameters.learning_rate,
        dataset_path=Path(cfg.paths.dataset_path),
    )

    dm = MFCCDataModule(
        dataset_path=Path(cfg.paths.dataset_path),
        batch_size=cfg.hyperparameters.batch_size,
        num_workers=cfg.params.num_workers,
        validation_size=cfg.params.validation_size,
        test_size=cfg.params.test_size,
    )

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
        max_epochs=cfg.hyperparameters.num_epochs,
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
            artifact_path=cfg.paths.artifact_path,
            conda_env=cfg.paths.conda_path,
            code_paths=[cfg.paths.code_path],
            signature=signature,
            registered_model_name=cfg.params.model_name,
            await_registration_for=0,
        )


if __name__ == "__main__":
    main()
