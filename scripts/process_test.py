from pathlib import Path
import pandas as pd
from src.gesture_from_path import gesture_from_path

import matplotlib.pyplot as plt
from src.kalman import kalman_filter

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from pathlib import Path
import shutil
import numpy as np

from config import (
    RAW_TRAIN_DIR,
    RAW_VAL_DIR,
    RAW_TEST_DIR,
    PROCESSED_TRAIN_DIR,
    PROCESSED_VAL_DIR,
    PROCESSED_TEST_DIR,
    MODELS_DIR,
    TRAINING_LOG_PATH,
    TRAIN_VAL_SPLIT
)