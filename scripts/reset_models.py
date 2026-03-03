# run this script if you want to reset all of the stored models and partitioned training and validation datasets
import shutil
from config import (
    RAW_TRAIN_DIR,
    RAW_VAL_DIR,
    PROCESSED_TRAIN_DIR,
    PROCESSED_VAL_DIR,
    MODELS_DIR,
    TRAINING_LOG_PATH
)

def empty_dir(directory):
    for item in directory.iterdir():
        item.unlink() # this assumes all items in the directory are files

def main():
    # take user input, are you sure you want to restart?
    user_input = input("Are you sure you want to reset all models and partitioned datasets? (y/n)")
    if user_input != "y":
        print("Aborting.")
        return
    
    # delete models
    empty_dir(MODELS_DIR)

    # delete training logs
    empty_dir(TRAINING_LOG_PATH)
    
    # delete partitioned datasets
    empty_dir(PROCESSED_TRAIN_DIR)
    empty_dir(PROCESSED_VAL_DIR)
    
    # move all raw validation data back into training set for repartitioning later
    if not any(RAW_VAL_DIR.iterdir()):
        print(f"Validation Directory is Empty Already! No need to move back to raw training dir")
    else:
        for txt_path in RAW_VAL_DIR.glob("*.txt"):
            shutil.move(txt_path, RAW_TRAIN_DIR / txt_path.name)
            
    print("All models and partitioned datasets deleted. Training dataset whole again.")

if __name__ == "__main__":
    main()