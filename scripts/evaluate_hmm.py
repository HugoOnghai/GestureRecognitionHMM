from train_hmm import load_models, train_and_save_models
from src.HMM.classify import classify
from src.load_seqs_by_label import load_seqs_by_label
from src.gesture import Gesture
from pathlib import Path
from config import (
    PROCESSED_TEST_DIR,
    PROCESSED_VAL_DIR,
    PROCESSED_TRAIN_DIR,
)
from collections import defaultdict
import numpy as np

def main():
    try:
        models = load_models()
    except:
        print("No models found in ./models/ — training models.")
        models = train_and_save_models()

    model_num_correct = 0
    model_num_total = 0

    # this will eventually change when we have the test set!
    processed_train_dir = Path("data/processed_train/")
    processed_val_dir = Path("data/processed_val/")

    seqs_by_label = load_seqs_by_label(processed_train_dir)

    ### TRAINING SET
    for label, seqs in seqs_by_label.items():
        for seq in seqs:
            predicted_label, prob = classify(seq, models)
            msg = f"True label: {label}, Predicted label: {predicted_label}, Probability: {prob}"

            model_num_total += 1
            if predicted_label == label:
                print(msg + " CORRECT!")
                model_num_correct += 1
            else:
                print(msg + " WRONG!")

    print(f"TRAINING 6-HMM accuracy: {model_num_correct / model_num_total}")

    ### VALIDATION SET
    if not any(processed_val_dir.iterdir()):
        print(f"Validation set was not tested since {processed_val_dir} was empty.")
    else:
        seqs_by_label = load_seqs_by_label(processed_val_dir)
        model_val_total = 0
        model_val_correct = 0
        for label, seqs in seqs_by_label.items():
            for seq in seqs:
                predicted_label, prob = classify(seq, models)
                msg = f"True label: {label}, Predicted label: {predicted_label}, Probability: {prob}"

                model_val_total += 1
                if predicted_label == label:
                    print(msg + " CORRECT!")
                    model_val_correct += 1
                else:
                    print(msg + " WRONG!")

    print(f"VALIDATION 6-HMM accuracy: {model_val_correct / model_val_total}")

    ### TEST SET
    if not any(PROCESSED_TEST_DIR.iterdir()):
        print(f"Test set was not tested since {PROCESSED_TEST_DIR} was empty.")
    else:
        seqs_by_label = defaultdict(list)

        for p in sorted(PROCESSED_TEST_DIR.glob("*.npz")):
            sample = np.load(p, allow_pickle=False)
            O = sample["O"] # my sequence of observations
            y = p.stem # get gesture type label, which is retrieved from the .npz file as an np.ndarray
            if O.size == 0:
                continue # in case I have an empty .npz file for some reason
            seqs_by_label[y].append(O)

        model_test_total = 0
        model_test_correct = 0
        for label, seqs in seqs_by_label.items():
            for seq in seqs:
                predicted_label, prob = classify(seq, models)
                msg = f"True label: {label}, Predicted label: {Gesture(predicted_label).name}, Probability: {prob}"

                model_test_total += 1
                if predicted_label == label:
                    print(msg + " CORRECT!")
                    model_test_correct += 1
                else:
                    print(msg)

    print(f"TEST 6-HMM accuracy: {model_test_correct / model_test_total}")

if __name__ == "__main__":
    main()
    