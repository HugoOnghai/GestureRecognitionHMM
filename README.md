# ECE 5242 Project 2: Gesture Recognition
## Hugo Onghai (NetID: hpo8)

# Introduction

In this project, I implemented six hidden markov models (HMMs) to each recognize one of the six possible gestures described by IMU data. The data was first filtered with a 1D Kalman Filter on each column (except time/ts). Then, the six HMMs employed the Baum-Welch expectation maximization algorithm to fit the training data provided.

# Code Structure

You can call `ls -T1` to recreate this diagram. The project is broken down into several sections, each of which is described here.

-  data # NOT PUSHED TO REMOTE REPOSITORY. It contains raw data and processed data (after Kalman Filtering)
-  models # TRAINED MODELS STORED HERE. Created/Loaded in `train_hmm.py` and `evaluate_hmm.py`
-  models # MODELS TRAINED ON FULL TRAINING SET. aka 0 weight fraction validation set
-  outputs # FIGURES + LOGS. Loglikelihood graph and 1D Kalman filter example.
-  scripts # RUN `evaluate_hmm.py`. This is where everything is put together, where predictions and accuracy (across all 6 HMMs treated as 1 model) are calcualted.
- 󰣞 src # KALMAN FILTER + HMM IMPLEMENTATION. Unlike last project, I organized all my code better as python modules.
-  tests # DEV TESTING. This has a jupyter notebook I made to test my src functions.
-  ECE5242_Project2.pdf # Project guidelines
-  pyproject.toml # Used by UV to manage the project.
- 󰂺 README.md # You are here!
-  uv.lock # Used by UV to synchronize dependencies.

# Setup Guide:

Like last time, I used UV again. After cloning the repository, run the following command at the project root:

```bash
uv run ./scripts/evaluate_hmm.py
```

This command should run `evaluate_hmm.py` in the UV virtual environment that will be built from `uv.lock`. It should load each of the HMMs, and then predict the labels of each sequence in the directory given, currently the processed training set.

# Debug/Testing:

Improved from last time, I implemented a validation set. This can be easily generated with `partition_preprocess.py`. Then, to train the models on the training set and evaluate them on the validation set, run `train_hmm.py` and `evaluate_hmm.py` respectively. If you want to reset everything and try a new partition, run `reset_models.py`. This script resets everything by deleting the contents of `/models/`, deleting all pre-processing (`/data/processed_train` and `/data/processed_val`), and deleting all logs (`/outputs/training_logs`).