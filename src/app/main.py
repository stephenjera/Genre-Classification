import logging
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import dagshub
import magic
import torch
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from starlette.templating import _TemplateResponse

sys.path.append(str(Path.cwd().parent))
from config_file import AppConfig
from database import SessionLocal, UploadedFile
from ml_model import MLModel

from genre_classifier.model import MFCCDataModule
from genre_classifier.preprocessing import save_mfcc

# Initialize app and config
app = FastAPI()
config = AppConfig()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Genre Classifier")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize ML model
ml_model = MLModel(config.MODEL_URI)


def get_db() -> Generator[Session, Any, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# @app.on_event("startup")
# async def startup_event():
# Initialize DagsHub
dagshub.init(
    repo_owner=config.DAGSHUB_OWNER,
    repo_name=config.DAGSHUB_REPO,
    mlflow=True,
)


@app.get(path="/", response_class=HTMLResponse)
async def home(request: Request) -> _TemplateResponse:
    return templates.TemplateResponse(name="index.html", context={"request": request})


@app.post(path="/predict")
async def predict(
    file: UploadFile = File(default=...), db: Session = Depends(dependency=get_db)
) -> JSONResponse:
    try:
        await validate_file(file=file)
        prediction = await process_file(file=file, db=db)
        return JSONResponse(content={"prediction": prediction})
    except HTTPException as e:
        return JSONResponse(content={"error": str(e)}, status_code=e.status_code)
    except Exception as e:
        logger.error(msg=f"Unexpected error during prediction: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)


async def validate_file(file: UploadFile) -> None:
    file_content = await file.read()
    file_type = magic.from_buffer(buffer=file_content, mime=True)
    logger.info(msg=f"File type: {file_type}")
    if not file_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    await file.seek(offset=0)


async def process_file(file: UploadFile, db: Session) -> str:
    with tempfile.TemporaryDirectory(dir=config.TEMP_DIR) as temp_dir:

        # temp_file_path = Path(config.TEMP_DIR) / file.filename # type: ignore
        temp_file_path = Path(temp_dir) / file.filename  # type: ignore
        temp_json_path = config.TEMP_DIR / "mfcc_data.json"
        logger.info(msg=f"uploaded file: {file.filename}")
        logger.info(msg=f"temp_file_path: {temp_file_path}")
        with temp_file_path.open(mode="wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        save_mfcc(
            dataset_path=config.TEMP_DIR, json_path=temp_json_path, **config.MFCC_CONFIG
        )

        X, _, _ = MFCCDataModule.load_data(dataset_path=temp_json_path)
        X2 = torch.tensor(data=X, dtype=torch.float32).clone().detach()
        predictions = ml_model.predict(input_tensor=X2)

        predicted_class_index = predictions.argmax().item()
        predicted_genre = config.MAPPINGS[str(predicted_class_index)]

        await save_file(
            file=file,
            temp_file_path=temp_file_path,
            predicted_genre=predicted_genre,
            db=db,
        )

        logger.info(f"Predicted: {predicted_genre}")
        return predicted_genre


async def save_file(
    file: UploadFile, temp_file_path: str | Path, predicted_genre: str, db: Session
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_filename = f"{timestamp}_{file.filename}"
    save_path = config.STORAGE_DIR / save_filename
    shutil.copy(src=temp_file_path, dst=save_path)

    new_file = UploadedFile(
        filename=file.filename,
        predicted_genre=predicted_genre,
        timestamp=datetime.now(),
        file_path=str(save_path),
    )
    db.add(instance=new_file)
    db.commit()


if __name__ == "__main__":
    uvicorn.run(app=app, host=config.HOST, port=config.PORT)
