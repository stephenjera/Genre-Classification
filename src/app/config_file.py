# Configuration
import os
from pathlib import Path
import dagshub


class AppConfig:
    _instance = None

    # Directories
    BASE_DIR = Path.cwd()  # Path(__file__).resolve().parent
    STORAGE_DIR = BASE_DIR / "uploaded_files"
    TEMP_DIR = BASE_DIR / "temp"

    # Ensure directories exist
    STORAGE_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

    # ML Model
    MODEL_URI = "models:/genre-classifier/8"

    # MFCC Configuration
    MFCC_CONFIG: dict[str, int] = {
        "samples_per_track": 22050,
        "n_mfcc": 13,
        "n_fft": 2048,
        "hop_length": 512,
        "num_segments": 1,
    }

    # Genre mappings
    MAPPINGS: dict[str, str] = {
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

    # DagsHub configuration
    DAGSHUB_OWNER = "stephenjera"
    DAGSHUB_REPO = "Genre-Classification"

    # Server configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))

    # MLflow
    MLFLOW_TRACKING_URI = "https://dagshub.com/stephenjera/Genre-Classification.mlflow"

    def __init__(self) -> None:
        os.environ["MLFLOW_TRACKING_URI"] = self.MLFLOW_TRACKING_URI
   

    @staticmethod
    def get_instance() -> "AppConfig":
        if AppConfig._instance is None:
            AppConfig._instance = AppConfig()
        return AppConfig._instance
