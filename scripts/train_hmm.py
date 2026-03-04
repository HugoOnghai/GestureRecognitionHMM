# important!! run this from the project root
# a cleaned-up version of implement_HMM.ipynb

import sys
from pathlib import Path
import os

project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.HMM.HMM import HMM
from src.gesture import Gesture
from src.load_seqs_by_label import load_seqs_by_label

PROCESSED_TRAIN_DIR = project_root / "data" / "processed_train"
PROCESSED_VAL_DIR = project_root / "data" / "processed_val"
MODELS_DIR = project_root / "models"
TRAINING_LOG_PATH = project_root / "outputs" / "training_logs" / "HMMs.txt"

NUM_STATES = 10
NUM_CLUSTERS = 100
MAX_ITERS = 10

def model_path_for_label(label: int) -> Path:
    """Return the .npz file path for a given gesture label."""
    name = Gesture(label).name  # one of the 6 gestures, 'WAVE', 'BEAT3' etc..
    return MODELS_DIR / f"{name}_HMM.npz"


def models_exist() -> bool:
    """Return True if ./models/ already contains at least one .npz file."""
    return any(MODELS_DIR.glob("*.npz"))


def load_models() -> dict:
    """Load all HMM .npz files from ./models/ and return a dict {label: HMM}."""
    if not models_exist():
        raise FileNotFoundError("No models found in ./models/. First run train_hmm.py to train models.")
    models = {}
    for p in sorted(MODELS_DIR.glob("*_HMM.npz")):
        hmm = HMM.load(p)
        models[hmm.label] = hmm
        print(f"Loaded model from {p.name}  (label={hmm.label}, "
              f"gesture={Gesture(hmm.label).name})")
    return models


def train_and_save_models() -> dict:
    """Train one HMM per gesture type, save each to ./models/, return dict."""
    seqs_by_label = load_seqs_by_label(PROCESSED_TRAIN_DIR)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = {}
    loglikelihoods = {}
    log_lines = []

    for label, seqs in seqs_by_label.items():
        hmm = HMM(N=NUM_STATES, M=NUM_CLUSTERS, init="random", label=label)
        ll_list = []

        msg = f"Now training HMM for {label}"
        print(msg)
        log_lines.append(msg)

        for epoch in range(MAX_ITERS):
            ll, termination_prob = hmm.fit_once(seqs)
            msg = (f"Epoch {epoch} complete. Log-likelihood = {ll}, "
                   f"Final P(seq|model) = {termination_prob}.")
            print(msg)
            log_lines.append(msg)
            ll_list.append(ll)

        models[label] = hmm
        loglikelihoods[label] = ll_list

        # save model
        out = model_path_for_label(label)
        hmm.save(out)
        print(f" --> Saved to {out.name}")

    # write training log
    TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_LOG_PATH.write_text("\n".join(log_lines) + "\n")
    print(f"\nTraining log written to {TRAINING_LOG_PATH}")

    return models

def main():
    if models_exist():
        print("Models found in ./models/ — loading from disk (training skipped).\n")
        models = load_models()
    else:
        print("No models found in ./models/ — starting training.\n")
        models = train_and_save_models()

    print(f"\n{len(models)} model(s) ready.")


if __name__ == "__main__":
    main()
