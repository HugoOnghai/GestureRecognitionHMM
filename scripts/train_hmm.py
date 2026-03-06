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

import matplotlib
matplotlib.use("Agg") # this prevent the gui from popping up when plots are made. just write directly to file
import matplotlib.pyplot as plt

from config import (
    PROCESSED_TEST_DIR,
    PROCESSED_TRAIN_DIR,
    PROCESSED_VAL_DIR,
    MODELS_DIR,
    TRAINING_LOG_PATH,
    FIGURE_PATH,
    EPSILON_LARGE,
    EPSILON_SMALL,
    NUM_CLUSTERS,
    NUM_STATES,
    MAX_ITERS
)

import argparse

# def parse_args():    
#     p = argparse.ArgumentParser()
#     p.add_argument("--train-val-split", type=float, default=0.2)
#     p.add_argument("--num-states", type=int, default=10)
#     p.add_argument("--num-clusters", type=int,default=100)
#     p.add_argument("--max-iters", type=int, default=100)
#     return p.parse_args()

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

    A = 0.5*np.eye(NUM_STATES)
    for i in range(NUM_STATES):
        if i == NUM_STATES - 1:
            A[i, 0] = 0.5
        else:
            A[i, i+1] = 0.5

    fig, axes = plt.subplots(1,2, figsize=(10,5))
    for label, seqs in seqs_by_label.items():
        hmm = HMM(N=NUM_STATES, M=NUM_CLUSTERS, A=A, init="random", label=label)
        ll_list = []

        msg = f"Now training HMM for {label}"
        print(msg)
        log_lines.append(msg)

        prev_ll = None
        for epoch in range(MAX_ITERS):
            ll = hmm.fit_once(seqs)
            ll_after_update = hmm.score(seqs)
            msg = (f"Epoch {epoch} complete. Log-likelihood = {ll_after_update}")
            print(msg)
            
            if prev_ll is not None and np.abs(ll - prev_ll) < EPSILON_LARGE:
                print("Convergence threshold met! Stopping training early.")
                break

            prev_ll = ll
            
            log_lines.append(msg)
            ll_list.append(ll_after_update)

        models[label] = hmm
        loglikelihoods[label] = ll_list

        axes[0].plot(range(1, len(loglikelihoods[label]) + 1), loglikelihoods[label], label=label)
        axes[0].set_xscale("symlog")
        axes[0].set_yscale("linear")
        min_len = min(len(v) for v in loglikelihoods.values())
        axes[1].plot(range(1, len(loglikelihoods[label]) + 1), loglikelihoods[label], label=label)
        axes[1].set_xlim(1, min_len)

        # save model
        out = model_path_for_label(label)
        hmm.save(out)
        print(f" --> Saved to {out.name}")

    # write training log
    TRAINING_LOG_FILEPATH = TRAINING_LOG_PATH / f"training_log"
    TRAINING_LOG_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_LOG_FILEPATH.write_text("\n".join(log_lines) + "\n")
    print(f"\nTraining log written to {TRAINING_LOG_PATH}")

    # write fig to png
    for ax in axes:
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Log-likelihood")
        ax.set_title("Training Plots")

    fig.tight_layout()
    fig.savefig(FIGURE_PATH / f"trainingplot_N_{NUM_STATES}_M_{NUM_CLUSTERS}.png")

    return models

def main():
    # # making it easier to hyperparameter tune with my bash script
    # args = parse_args()
    # NUM_STATES = args.num_states
    # NUM_CLUSTERS = args.num_clusters
    # MAX_ITERS = args.max_iters
    # TRAIN_VAL_SPLIT = args.train_val_split

    if models_exist():
        print("Models found in ./models/ — loading from disk (training skipped).\n")
        models = load_models()
    else:
        print("No models found in ./models/ — starting training.\n")
        models = train_and_save_models()

    print(f"\n{len(models)} model(s) ready.")


if __name__ == "__main__":
    main()
