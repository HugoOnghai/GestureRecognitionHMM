from pathlib import Path
from src.gesture import Gesture

def gesture_from_path(p: Path) -> str:
    stem = Path(p).stem.lower()
    if stem.startswith("beat4"):
        return Gesture['BEAT4']
    if stem.startswith("beat3"):
        return Gesture['BEAT3']
    if stem.startswith("eight"):
        return Gesture['EIGHT']
    if stem.startswith("circle"):
        return Gesture['CIRCLE']
    if stem.startswith("wave"):
        return Gesture['WAVE']
    if stem.startswith("inf"):
        return Gesture['INFINITY']
    raise ValueError(f"Unknown gesture in filename: {stem}")