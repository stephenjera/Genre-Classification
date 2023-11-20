import pytorch_lightning as pl
import mlflow
import dagshub
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from hydra.core.config_store import ConfigStore
from config import GenreClassifierConfig
from genre_classifier.model import LSTMGenreModel, MFCCDataModule
from pathlib import Path
from mlflow.models.signature import infer_signature


cs = ConfigStore.instance()
cs.store(name="genre_config", node=GenreClassifierConfig)


def save_confusion_matrix(
    model: LSTMGenreModel, mappings: dict[str, str], filename: str
):
    confusion_matrix = model.confusion_matrix.compute()
    cm = confusion_matrix.detach().cpu().numpy()

    _, ax = plt.subplots(figsize=(10, 10))

    # Set tick labels from mappings
    labels = [v for _, v in sorted(mappings.items(), key=lambda item: int(item[0]))]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=labels,  # type: ignore
        yticklabels=labels,  # type: ignore
    )

    plt.title("Genre Classifier Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    save_path = f"{filename}.png"
    plt.savefig(save_path)
    return save_path


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

    dagshub.init(
        repo_owner="stephenjera",
        repo_name="Genre-Classification",
        mlflow=True,
    )
    mlflow.pytorch.autolog(log_models=False)

    trainer = pl.Trainer(
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
        _, _, mappings = dm.load_data(cfg.paths.dataset_path)
        mlflow.log_artifact(
            save_confusion_matrix(
                model=model, mappings=mappings, filename="confusion_matrix"
            )
        )


if __name__ == "__main__":
    main()
