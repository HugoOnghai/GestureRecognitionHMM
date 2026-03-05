# this is pretty much the same as /tests/visualize_quantize.ipynb which I made first
# except that this is the cleaned up version of it for convenience

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

### SPLIT VAL AND TRAIN
source = RAW_TRAIN_DIR
destination = RAW_VAL_DIR

# train_val_split now specified in config
if not any(destination.iterdir()):
    print(f"Validation Directory is Empty, Moving {TRAIN_VAL_SPLIT*100}% of Training Set to Validation Set")
    for txt_path in source.glob("*.txt"):
        moveToVal = np.random.choice(a=[True, False], p=[TRAIN_VAL_SPLIT, 1-TRAIN_VAL_SPLIT])
        if moveToVal:
            dest = shutil.move(txt_path, destination / txt_path.name)
    print(f"Partition complete.")
else:
    print("Validation Directory Not Empty, Restore Training Set Before Partitioning")

### PROCESSING RAW TRAINING AND VAL DATA

from pathlib import Path
import pandas as pd
from src.gesture_from_path import gesture_from_path

import matplotlib.pyplot as plt
from src.kalman import kalman_filter

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

col_names = ["ts", "Wx", "Wy", "Wz", "Ax", "Ay", "Az"]
sensor_cols = ["Wx", "Wy", "Wz", "Ax", "Ay", "Az"]

X_chunks = []
records = []

### TRAINING DATA FIT + TRANSFORM
for txt_path in RAW_TRAIN_DIR.glob("*.txt"):
    # import raw training data
    raw_gesture = pd.read_csv(txt_path, header=None, sep=r"\s+", engine="python") # used regex to separate cols by any whitespace
    raw_gesture.columns = col_names

    # get gesture type
    gesture_type = gesture_from_path(txt_path)

    # apply 1D kalman filter to each column of IMU data
    ts = raw_gesture["ts"].to_numpy()
    fil_gesture = pd.DataFrame({"ts": ts})
    for col in sensor_cols:
        fil_gesture[col] = kalman_filter(raw_gesture[col].to_numpy(dtype="float64"))

    X = fil_gesture[sensor_cols].to_numpy(dtype=np.float32)
    X_chunks.append(X)

    # make record for final formatting later
    records.append(
        {
            "id": txt_path.stem,
            "gesture": gesture_type,
            "ts": ts,
            "X": X  
        }
    )

X_all = np.vstack(X_chunks) # merge all "per file" data points to one large numpy array

# scale all sensor data
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# fit K-means clustering to discretize observation space
num_clusters = 100
kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(X_all_scaled)

# format all processed data ready for HMM
processed_dir = PROCESSED_TRAIN_DIR
processed_dir.mkdir(parents=True, exist_ok=True)

for record in records:
    # since we fit k-means onto scaled data, we must scale our data again
    X_scaled = scaler.transform(record["X"])
    O = kmeans.predict(X_scaled) # observations

    out_path = processed_dir / f"{record["id"]}.npz"
    np.savez(
        out_path,
        O = O,
        gesture = record["gesture"],
        ts = record["ts"],
        id = record["id"],
        num_clusters = num_clusters # store the number of clusters
    )

### VALIDATION SET TRANSFORM
if not any(RAW_VAL_DIR.iterdir()):
    print("No samples allocated to the validation set, skipping validation set preprocessing!")
else:
    col_names = ["ts", "Wx", "Wy", "Wz", "Ax", "Ay", "Az"]
    sensor_cols = ["Wx", "Wy", "Wz", "Ax", "Ay", "Az"]

    X_chunks = []
    records = []

    for txt_path in RAW_VAL_DIR.glob("*.txt"):
        # import raw training data
        raw_gesture = pd.read_csv(txt_path, header=None, sep=r"\s+", engine="python") # used regex to separate cols by any whitespace
        raw_gesture.columns = col_names

        # get gesture type
        gesture_type = gesture_from_path(txt_path)

        # apply 1D kalman filter to each column of IMU data
        ts = raw_gesture["ts"].to_numpy()
        fil_gesture = pd.DataFrame({"ts": ts})
        for col in sensor_cols:
            fil_gesture[col] = kalman_filter(raw_gesture[col].to_numpy(dtype="float64"))

        X = fil_gesture[sensor_cols].to_numpy(dtype=np.float32)
        X_chunks.append(X)

        # make record for final formatting later
        records.append(
            {
                "id": txt_path.stem,
                "gesture": gesture_type,
                "ts": ts,
                "X": X  
            }
        )

    X_val = np.vstack(X_chunks) # merge all "per file" data points to one large numpy array

    # scale all sensor data (DONT FIT VALIDATION DATA)
    X_val_scaled = scaler.transform(X_val)

    # kmeans, don't fit just transform
    # format all processed data ready for HMM
    processed_val_dir = PROCESSED_VAL_DIR
    processed_val_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        # since we fit k-means onto scaled data, we must scale our data again
        X_scaled = scaler.transform(record["X"])
        O = kmeans.predict(X_scaled) # observations

        out_path = processed_val_dir / f"{record["id"]}.npz"
        np.savez(
            out_path,
            O = O,
            gesture = record["gesture"],
            ts = record["ts"],
            id = record["id"],
            num_clusters = num_clusters # store the number of clusters
        )

### TEST SET

### TEST SET TRANSFORM
if not any(RAW_TEST_DIR.iterdir()):
    print("No samples allocated to the test set, skipping test set preprocessing!")
else:
    col_names = ["ts", "Wx", "Wy", "Wz", "Ax", "Ay", "Az"]
    sensor_cols = ["Wx", "Wy", "Wz", "Ax", "Ay", "Az"]

    X_chunks = []
    records = []

    for txt_path in RAW_TEST_DIR.glob("*.txt"):
        # import raw training data
        raw_gesture = pd.read_csv(txt_path, header=None, sep=r"\s+", engine="python") # used regex to separate cols by any whitespace
        raw_gesture.columns = col_names

        # get gesture type
        gesture_type = None

        # apply 1D kalman filter to each column of IMU data
        ts = raw_gesture["ts"].to_numpy()
        fil_gesture = pd.DataFrame({"ts": ts})
        for col in sensor_cols:
            fil_gesture[col] = kalman_filter(raw_gesture[col].to_numpy(dtype="float64"))

        X = fil_gesture[sensor_cols].to_numpy(dtype=np.float32)
        X_chunks.append(X)

        # make record for final formatting later
        records.append(
            {
                "id": txt_path.stem,
                "gesture": gesture_type,
                "ts": ts,
                "X": X  
            }
        )

    X_test = np.vstack(X_chunks) # merge all "per file" data points to one large numpy array

    # scale all sensor data (DONT FIT TEST DATA)
    X_test_scaled = scaler.transform(X_test)

    # kmeans, don't fit just transform
    # format all processed data ready for HMM
    processed_test_dir = PROCESSED_TEST_DIR
    processed_test_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        # since we fit k-means onto scaled data, we must scale our data again
        X_scaled = scaler.transform(record["X"])
        O = kmeans.predict(X_scaled) # observations

        out_path = processed_test_dir / f"{record["id"]}.npz"
        np.savez(
            out_path,
            O = O,
            gesture = record["gesture"],
            ts = record["ts"],
            id = record["id"],
            num_clusters = num_clusters # store the number of clusters
        )