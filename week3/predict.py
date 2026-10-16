# -*- coding: utf-8 -*-
"""
Week 3 - single-image inference.

Loads a saved checkpoint and classifies one photo, returning the category and the
model's confidence. This module is the inference backend for both the command-line
tool and the Flask web app in ``app.py``.


"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Quieten TensorFlow's routine notices before it is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image

WEEK3_DIR = Path(__file__).resolve().parent
WEEK1_DIR = WEEK3_DIR.parent / "week1"

if str(WEEK1_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK1_DIR))

import config  # noqa: E402  - Week 1's config: class names and input size
from utils import silence_tensorflow  # noqa: E402

silence_tensorflow()

# --------------------------------------------------------------------------- model choice
# Checked in order. The fine-tuned checkpoints are preferred only because they are
# the most recent training - on the 379-image test set the three models are
# statistically indistinguishable (see week2 §9), so this is a deployment
# convenience, not a claim that one of them is better.
CANDIDATE_CHECKPOINTS = [
    WEEK3_DIR.parent / "week2" / "outputs" / "finetuned_model.keras",
    WEEK3_DIR.parent / "week2" / "outputs" / "finetuned_classweight_model.keras",
    WEEK1_DIR / "outputs" / "baseline_model.keras",
]

DEFAULT_TOP_K = 3

_model = None
_model_path: Path | None = None


def resolve_checkpoint(explicit: str | Path | None = None) -> Path:
    """Choose which checkpoint to serve: explicit path, then $WASTE_MODEL, then defaults."""
    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("WASTE_MODEL")
    if env:
        candidates.append(Path(env))

    if candidates:
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            "Checkpoint not found: " + ", ".join(str(p) for p in candidates)
        )

    for path in CANDIDATE_CHECKPOINTS:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No trained checkpoint found. Looked for:\n  "
        + "\n  ".join(str(p) for p in CANDIDATE_CHECKPOINTS)
        + "\n\nRun week1/waste_classification_week1.ipynb first (and week2's notebook "
        "if you want the fine-tuned models)."
    )


def load_model(checkpoint: str | Path | None = None):
    """
    Load the Keras model, memoised.

    Loading takes a few seconds, so the web app does it once at startup rather than
    once per request. Returns ``(model, checkpoint_path)``.
    """
    global _model, _model_path

    path = resolve_checkpoint(checkpoint)
    if _model is None or _model_path != path:
        import tensorflow as tf

        _model = tf.keras.models.load_model(path)
        _model_path = path
    return _model, _model_path


def preprocess(image: Image.Image) -> np.ndarray:
    """
    Turn a PIL image into the array the model expects.

    The saved model already contains the MobileNetV2 rescaling *and* the augmentation
    layers (which are inert outside training), so inference feeds raw 0-255 pixels.
    Normalising here as well would apply the transformation twice - the classic way
    to silently degrade a model after deployment.
    """
    image = image.convert("RGB").resize(config.IMG_SIZE, Image.BILINEAR)
    array = np.asarray(image, dtype="float32")     # 0-255, exactly as during training
    return array[None, ...]                        # add the batch dimension


def predict_array(batch: np.ndarray, model=None, top_k: int = DEFAULT_TOP_K) -> dict:
    """Classify a preprocessed batch and return labels with confidences."""
    if model is None:
        model = load_model()[0]

    probabilities = model.predict(batch, verbose=0)[0]
    order = np.argsort(-probabilities)[:top_k]

    return {
        "label": config.CLASS_NAMES[int(order[0])],
        "confidence": float(probabilities[order[0]]),
        "top_k": [
            {"label": config.CLASS_NAMES[int(i)], "confidence": float(probabilities[i])}
            for i in order
        ],
        "all_probabilities": {
            name: float(p) for name, p in zip(config.CLASS_NAMES, probabilities)
        },
    }


def predict_image(image: Image.Image, model=None, top_k: int = DEFAULT_TOP_K) -> dict:
    """Classify a PIL image. The function the web app calls."""
    return predict_array(preprocess(image), model=model, top_k=top_k)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify a single waste photo.")
    parser.add_argument("image", nargs="?", help="path to the image file")
    parser.add_argument("--image", dest="image_option", help="path to the image file")
    parser.add_argument("--model", help="path to a .keras checkpoint")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args(argv)

    path = args.image or args.image_option
    if not path:
        parser.error("give a path to an image, for example:  python predict.py photo.jpg")

    model, checkpoint = load_model(args.model)
    with Image.open(path) as image:
        result = predict_image(image, model=model, top_k=args.top_k)

    print(f"checkpoint : {checkpoint}")
    print(f"image      : {path}")
    print(f"prediction : {result['label']}  ({result['confidence']:.1%} confident)")
    print("top-k      :")
    for item in result["top_k"]:
        print(f"  {item['label']:<10} {item['confidence']:6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
