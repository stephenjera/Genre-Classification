import mlflow
import torch

class MLModel:
    def __init__(self, model_uri: str):
        self.model = mlflow.pytorch.load_model(model_uri)

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model.predict_step(input_tensor)