# this module allows me to import file names consistent across all my scripts.

import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.insert(0, str(PROJECT_ROOT))

def _get(name, cast, default=None): # so i can change hyperparameters directly from the bash script
    val = os.getenv(name)
    if val is None:
        if default is None:
            raise RuntimeError(f"{name} not set")
        return default
    return cast(val)

TRAIN_VAL_SPLIT = _get("TRAIN_VAL_SPLIT", float, 0.2) # this is the percent change any given file goes to the val set. DON'T SET THIS TOO LARGE SUCH THAT THE TRAINING SET CAN BE EMPTY, I DIDN'T ACCOUNT FOR THAT

NUM_STATES = _get("NUM_STATES", int, 10)
NUM_CLUSTERS = _get("NUM_CLUSTERS", int, 100)
MAX_ITERS = _get("MAX_ITERS", int, 100)

RAW_TRAIN_DIR = PROJECT_ROOT / "data" / "raw_train"
RAW_VAL_DIR = PROJECT_ROOT / "data" / "raw_val"
RAW_TEST_DIR = PROJECT_ROOT / "data" / "raw_test"

PROCESSED_TRAIN_DIR = PROJECT_ROOT / "data" / "processed_train"
PROCESSED_VAL_DIR = PROJECT_ROOT / "data" / "processed_val"
PROCESSED_TEST_DIR = PROJECT_ROOT / "data" / "processed_test"

MODELS_DIR = PROJECT_ROOT / "models"

TRAINING_LOG_PATH = PROJECT_ROOT / "outputs" / "training_logs"
TESTING_LOG_PATH = PROJECT_ROOT / "outputs" / "testing_logs"
FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures"

EPSILON_LARGE = 1e-4
EPSILON_SMALL = 1e-12

# Auto-create all directories so a fresh clone never hits "directory not found"
# since I don't push this to the repo. Whenever a script that imports the config gets run, this should run too.
for _dir in [
    RAW_TRAIN_DIR, RAW_VAL_DIR, RAW_TEST_DIR,
    PROCESSED_TRAIN_DIR, PROCESSED_VAL_DIR, PROCESSED_TEST_DIR,
    MODELS_DIR,
    TRAINING_LOG_PATH,
    TESTING_LOG_PATH,
    FIGURE_PATH
]:
    _dir.mkdir(parents=True, exist_ok=True)
