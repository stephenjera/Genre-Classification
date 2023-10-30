# Create the model
import pytorch_lightning as pl
import mlflow
import dagshub
from model import LSTMGenreModel
from pathlib import Path

# Hyperparameters
num_classes = 10  # number of genres
input_size = 13  # number of MFCC coefficients
hidden_size = 128
num_layers = 2
batch_size = 64
num_epochs = 20
learning_rate = 1e-3

dataset_path = Path.cwd() / "src" / "data.json"

model = LSTMGenreModel(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes,
    batch_size=batch_size,
    learning_rate=learning_rate,
    dataset_path=dataset_path,
)

dagshub.init(repo_owner="stephenjera", repo_name="Genre-Classification", mlflow=True)

mlflow.pytorch.autolog()

# Create the trainer
trainer = pl.Trainer(max_epochs=num_epochs, log_every_n_steps=25)

# Train the model
trainer.fit(model)

# Evaluate the model
trainer.validate(model)

# Save the model
trainer.save_checkpoint("checkpoint.ckpt")
