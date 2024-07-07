import json
import os
# Import your preprocessing function
import sys
import tempfile
from pathlib import Path
from typing import Union

import dagshub
import mlflow
import torch
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.templating import _TemplateResponse

sys.path.append(str(Path.cwd().parent))
from genre_classifier.model import MFCCDataModule
from genre_classifier.preprocessing import save_mfcc

app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Get the current directory
TEMP_DIR = Path.cwd() / "temp"
TEMP_DIR.mkdir(exist_ok=True)  # Create the temp directory if it doesn't exist

# Load your ML model here
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/stephenjera/Genre-Classification.mlflow"
dagshub.init(
    repo_owner="stephenjera",
    repo_name="Genre-Classification",
    mlflow=True,
)

model_uri = "models:/genre-classifier/8"
loaded_model = mlflow.pytorch.load_model(model_uri)

mappings = {
        "0": "blues",
        "1": "classical",
        "2": "country",
        "3": "disco",
        "4": "hiphop",
        "5": "jazz",
        "6": "metal",
        "7": "pop",
        "8": "reggae",
        "9": "rock"
    }


print(TEMP_DIR)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> _TemplateResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:

    temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    try:
        # Create a temporary directory to store the uploaded file
        # with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
        temp_file_path = Path(temp_dir) / file.filename # type: ignore
        temp_json_path = Path(temp_dir) / "mfcc_data.json"
        print(f"uploaded file: {file.filename}")
        print(f"temp_file_path: {temp_file_path}")

        # Save the uploaded file
        with temp_file_path.open("wb") as buffer:
            buffer.write(await file.read())

        # Create a temporary JSON file to store the MFCC data
        # temp_json_path = Path(temp_dir) / "mfcc_data.json"

        # Preprocess the audio file
        save_mfcc(
            dataset_path=TEMP_DIR,
            json_path=temp_json_path,
            samples_per_track=22050,  # Assuming 30-second clips at 22050 Hz
            n_mfcc=13,
            n_fft=2048,
            hop_length=512,
            num_segments=1,
        )

        # Load the preprocessed data
        # with temp_json_path.open() as f:
        #     mfcc_data = json.load(f)

        # Make prediction
        # Assuming your model expects the MFCC data in a specific format
        # You may need to adjust this part based on your model's requirements
        X, y, _ = MFCCDataModule.load_data(temp_json_path)
        X2 = torch.tensor(X, dtype=torch.float32).clone().detach()
        prediction = loaded_model.predict_step(X2[:1])

        predicted_class_index = prediction.argmax().item()
        print(
            f"prediction:{predicted_class_index}, {mappings[str(predicted_class_index)]} Actual {y[0]}, {mappings[str(y[0])]}"
        )

        # For now, we'll return a placeholder prediction
        prediction = "Placeholder prediction"
        prediction = mappings[str(predicted_class_index)]

        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
