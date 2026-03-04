# this module allows me to import file names consistent across all my scripts.

import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_TRAIN_DIR = PROJECT_ROOT / "data" / "raw_train"
RAW_VAL_DIR = PROJECT_ROOT / "data" / "raw_val"
RAW_TEST_DIR = PROJECT_ROOT / "data" / "raw_test"

PROCESSED_TRAIN_DIR = PROJECT_ROOT / "data" / "processed_train"
PROCESSED_VAL_DIR = PROJECT_ROOT / "data" / "processed_val"
PROCESSED_TEST_DIR = PROJECT_ROOT / "data" / "processed_test"

MODELS_DIR = PROJECT_ROOT / "models"

TRAINING_LOG_PATH = PROJECT_ROOT / "outputs" / "training_logs"

TRAIN_VAL_SPLIT = 0.2 # DON'T SET THIS TOO LARGE SUCH THAT THE TRAINING SET CAN BE EMPTY, I DIDN'T ACCOUNT FOR THAT

# Auto-create all directories so a fresh clone never hits "directory not found"
# since I don't push this to the repo. Whenever a script that imports the config gets run, this should run too.
for _dir in [
    RAW_TRAIN_DIR, RAW_VAL_DIR, RAW_TEST_DIR,
    PROCESSED_TRAIN_DIR, PROCESSED_VAL_DIR, PROCESSED_TEST_DIR,
    MODELS_DIR,
    TRAINING_LOG_PATH,
]:
    _dir.mkdir(parents=True, exist_ok=True)