from collections import defaultdict
from pathlib import Path
import numpy as np

def load_seqs_by_label(processed_dir: Path):
    seqs_by_label = defaultdict(list)

    for p in sorted(processed_dir.glob("*.npz")):
        sample = np.load(p, allow_pickle=False)
        O = sample["O"] # my sequence of observations
        y = int(sample["gesture"]) # get gesture type label, which is retrieved from the .npz file as an np.ndarray
        if O.size == 0:
            continue # in case I have an empty .npz file for some reason
        seqs_by_label[y].append(O) # add this sequence to the label group

    return seqs_by_label
        