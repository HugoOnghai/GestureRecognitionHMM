# this module allows me to import file names consistent across all my scripts.

import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_TRAIN_DIR = PROJECT_ROOT / "data" / "raw_train"
RAW_VAL_DIR = PROJECT_ROOT / "data" / "raw_val"

PROCESSED_TRAIN_DIR = PROJECT_ROOT / "data" / "processed_train"
PROCESSED_VAL_DIR = PROJECT_ROOT / "data" / "processed_val"

MODELS_DIR = PROJECT_ROOT / "models"

TRAINING_LOG_PATH = PROJECT_ROOT / "outputs" / "training_logs"

TRAIN_VAL_SPLIT = 0.8