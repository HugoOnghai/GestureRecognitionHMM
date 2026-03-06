from train_hmm import load_models, train_and_save_models
from src.HMM.classify import classify
from src.load_seqs_by_label import load_seqs_by_label
from src.gesture import Gesture
from pathlib import Path
from config import (
    PROCESSED_TEST_DIR,
    PROCESSED_VAL_DIR,
    PROCESSED_TRAIN_DIR,
    TESTING_LOG_PATH
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
    print("="*20, "Now evaluating the training set!", "="*20)
    for label, seqs in seqs_by_label.items():
        for seq in seqs:
            predicted_label, ll, _ = classify(seq, models)
            msg = f"True label: {label}, Predicted label: {predicted_label}, -Loglikelihood: {ll}"

            model_num_total += 1
            if predicted_label == label:
                print(msg + " CORRECT!")
                model_num_correct += 1
            else:
                print(msg + " WRONG!")

    print(f"TRAINING 6-HMM accuracy: {model_num_correct / model_num_total}")

    ### VALIDATION SET
    print("="*20, "Now evaluating the validation set!", "="*20)
    if not any(processed_val_dir.iterdir()):
        print(f"Validation set was not tested since {processed_val_dir} was empty.")
    else:
        seqs_by_label = load_seqs_by_label(processed_val_dir)
        model_val_total = 0
        model_val_correct = 0
        for label, seqs in seqs_by_label.items():
            for seq in seqs:
                predicted_label, ll, _ = classify(seq, models)
                msg = f"True label: {label}, Predicted label: {predicted_label}, -Loglikelihood: {ll}"

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
        print("="*20, "Now evaluating the testing set!", "="*20)
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

        log_msg = ""
        for label, seqs in seqs_by_label.items():
            for seq in seqs:
                predicted_label, predicted_ll, scores = classify(seq, models)

                msg = (
                    f"Prediction for {label}:\n"
                    f"Top #1 {Gesture(scores[0][0]).name:<10} loglikelihood = {scores[0][1]:.2f}\n"
                    f"Top #2 {Gesture(scores[1][0]).name:<10} loglikelihood = {scores[1][1]:.2f}\n"
                    f"Top #3 {Gesture(scores[2][0]).name:<10} loglikelihood = {scores[2][1]:.2f}\n"
                )

                print(msg)

                log_msg += f"Full model ranking for {label}:\n"
                for rank, (hmm_name, score) in enumerate(scores, start=1):
                    log_msg += f"{rank}. {Gesture(hmm_name).name:<10} loglikelihood = {score:.2f}\n"

        # write training log
        TESTING_LOG_FILEPATH = TESTING_LOG_PATH / f"testing_log"
        TESTING_LOG_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
        TESTING_LOG_FILEPATH.write_text(log_msg)

        print(f"Testing Complete! Log written to {TESTING_LOG_FILEPATH}")

if __name__ == "__main__":
    main()
    
