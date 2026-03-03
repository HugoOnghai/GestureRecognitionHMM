from train_hmm import load_models, train_and_save_models
from src.HMM.classify import classify
from src.load_seqs_by_label import load_seqs_by_label
from pathlib import Path

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
        Print(f"Validation set was not tested since {processed_val_dir} was empty.")
    else:
        seqs_by_label = load_seqs_by_label(processed_val_dir)
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

    print(f"VALIDATION 6-HMM accuracy: {model_num_correct / model_num_total}")

if __name__ == "__main__":
    main()
    