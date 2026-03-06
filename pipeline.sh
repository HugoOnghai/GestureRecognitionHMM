set -e

export NUM_STATES=10
export NUM_CLUSTERS=100
export MAX_ITERS=100
export TRAIN_VAL_SPLIT=0

uv run ./scripts/reset_models.py
uv run ./scripts/partition_preprocess.py
uv run ./scripts/train_hmm.py
uv run ./scripts/evaluate_hmm.py