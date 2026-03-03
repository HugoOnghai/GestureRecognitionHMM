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
    seqs_by_label = load_seqs_by_label(processed_train_dir)

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

    print(f"6-HMM accuracy: {model_num_correct / model_num_total}")

if __name__ == "__main__":
    main()
    