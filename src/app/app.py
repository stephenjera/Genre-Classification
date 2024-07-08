import logging
import os
import sys
import tempfile
from pathlib import Path

import dagshub
import magic
import mlflow
import torch
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.templating import _TemplateResponse

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import shutil
from datetime import datetime

sys.path.append(str(Path.cwd().parent))
from genre_classifier.model import MFCCDataModule
from genre_classifier.preprocessing import save_mfcc

STORAGE_DIR = Path.cwd() / "uploaded_files"
STORAGE_DIR.mkdir(exist_ok=True)

Base = declarative_base()
engine = create_engine('sqlite:///uploaded_files.db')
Session = sessionmaker(bind=engine)

class UploadedFile(Base):
    __tablename__ = 'uploaded_files'
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    predicted_genre = Column(String)
    timestamp = Column(String)
    file_path = Column(String)

Base.metadata.create_all(engine)


app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Genre Classifier")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

TEMP_DIR = Path.cwd() / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# Load ML model here
os.environ["MLFLOW_TRACKING_URI"] = (
    "https://dagshub.com/stephenjera/Genre-Classification.mlflow"
)
dagshub.init(
    repo_owner="stephenjera",
    repo_name="Genre-Classification",
    mlflow=True,
)

model_uri = "models:/genre-classifier/8"
loaded_model = mlflow.pytorch.load_model(model_uri)

# Configuration
MFCC_CONFIG = {
    "samples_per_track": 22050,
    "n_mfcc": 13,
    "n_fft": 2048,
    "hop_length": 512,
    "num_segments": 1,
}

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
    "9": "rock",
}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> _TemplateResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    # Check file type
    file_content = await file.read()
    file_type = magic.from_buffer(file_content, mime=True)
    logger.info(f"File type: {file_type}")
    if not file_type.startswith('audio/'):
        return JSONResponse(content={"error": "Invalid file type"}, status_code=400)

    with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
        try:
            # Create a temporary directory to store the uploaded file
            #temp_dir = tempfile.mkdtemp(dir=TEMP_DIR) # for debugging
            temp_file_path = Path(temp_dir) / file.filename  # type: ignore
            temp_json_path = Path(temp_dir) / "mfcc_data.json"
            logger.info(f"uploaded file: {file.filename}")
            logger.info(f"temp_file_path: {temp_file_path}")

            # Save the uploaded file
            with temp_file_path.open("wb") as buffer:
                # buffer.write(await file.read())
                buffer.write(file_content)

            # Preprocess the audio file
            save_mfcc(dataset_path=TEMP_DIR, json_path=temp_json_path, **MFCC_CONFIG)

            # Make prediction
            X, _, _ = MFCCDataModule.load_data(temp_json_path)
            X2 = torch.tensor(X, dtype=torch.float32).clone().detach()
            predictions = loaded_model.predict_step(X2)

            predicted_class_index = predictions.argmax().item()
            predicted_genre = mappings[str(predicted_class_index)]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"{timestamp}_{file.filename}"
            save_path = STORAGE_DIR / save_filename
            shutil.copy(temp_file_path, save_path)

            session = Session()
            new_file = UploadedFile(filename=file.filename, 
                                    predicted_genre=predicted_genre,
                                    timestamp=timestamp,
                                    file_path=str(save_path))
            session.add(new_file)
            session.commit()

            logger.info(f"Predicted: {predicted_genre}")
            return JSONResponse(content={"prediction": predicted_genre})

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            return JSONResponse(content={"error": str(e)}, status_code=400)


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
