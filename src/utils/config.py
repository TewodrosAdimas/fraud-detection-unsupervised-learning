from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2] 
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed"