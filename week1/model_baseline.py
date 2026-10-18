
"""
Week 1 - step 3: the baseline transfer-learning model.

Two-stage transfer-learning, stage 1: keep the ImageNet-pretrained MobileNetV2
convolutional base completely frozen and train only a small classification head.
This validates that the whole pipeline learns something before any expensive
fine-tuning is attempted (that is Week 2).

Architecture (matches the plan sent to the tutor):

    image (150x150x3, 0..255)
      -> Rescaling(1/127.5, -1)      # == mobilenet_v2.preprocess_input
      -> data augmentation           # active only while training
      -> MobileNetV2 base (frozen, training=False)
      -> GlobalAveragePooling2D      # far fewer parameters than Flatten
      -> Dropout(0.2)
      -> Dense(6, softmax)

"""

from __future__ import annotations

import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
from utils import header, human_bytes, silence_tensorflow

silence_tensorflow()

# Weight files MobileNetV2 needs for `weights="imagenet"`, in the exact cache
# layout Keras expects. Listed so the download can be pre-flighted and retried.
WEIGHTS_DIR = Path(os.path.expanduser("~")) / ".keras" / "models"
WEIGHTS_FILES = [
    f"mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_{config.IMG_SIZE[0]}_no_top.h5",
]
WEIGHTS_BASE_URL = "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/"


def build_data_augmentation() -> keras.Sequential:
    """Augmentation as model layers: automatically disabled at inference time."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )


def load_pretrained_base(trainable: bool = False) -> keras.Model:
    """MobileNetV2 without its ImageNet classifier head, frozen by default."""
    base = keras.applications.MobileNetV2(
        weights="imagenet",
        input_shape=(*config.IMG_SIZE, 3),
        include_top=False,
    )
    base.trainable = trainable
    return base


def build_baseline_model(
    *,
    num_classes: int = config.NUM_CLASSES,
    img_size: tuple[int, int] = config.IMG_SIZE,
    dropout: float = config.DROPOUT,
    learning_rate: float = config.LEARNING_RATE,
    base_model: keras.Model | None = None,
) -> tuple[keras.Model, keras.Model]:
    """Assemble and compile the baseline model. Returns ``(model, base_model)``."""
    if base_model is None:
        base_model = load_pretrained_base(trainable=False)
    base_model.trainable = False

    inputs = keras.Input(shape=(*img_size, 3), name="image")
    # Same maths as keras.applications.mobilenet_v2.preprocess_input, but saved
    # *inside* the model so inference never has to repeat the normalisation.
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inputs)
    x = build_data_augmentation()(x)
    # training=False keeps BatchNormalization in inference mode, protecting the
    # pretrained running statistics while the head is being trained.
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="waste_baseline_mobilenetv2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def describe_model(model: keras.Model, base_model: keras.Model | None = None) -> dict:
    """Return a small summary dict for the generated report."""
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    non_trainable = sum(int(tf.size(w)) for w in model.non_trainable_weights)
    return {
        "total_params": int(model.count_params()),
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "base_trainable": bool(base_model.trainable) if base_model is not None else None,
    }


def preflight_pretrained_weights() -> bool:
    """
    Check (and if needed download) the MobileNetV2 ImageNet weights.

    Keras fetches them from storage.googleapis.com, which is not reachable from
    every network. Calling this early turns an obscure mid-training failure into
    a clear message with a manual fix.
    """
    header("Pre-flight: MobileNetV2 ImageNet weights", "-")
    missing = [f for f in WEIGHTS_FILES if not (WEIGHTS_DIR / f).exists()]
    if not missing:
        for name in WEIGHTS_FILES:
            size = (WEIGHTS_DIR / name).stat().st_size
            print(f"  [cached] {name} ({human_bytes(size)})")
        return True

    print(f"  cache dir: {WEIGHTS_DIR}")
    print(f"  missing  : {', '.join(missing)}")
    print(f"  fetching from {WEIGHTS_BASE_URL} ...")

    try:
        from tensorflow.keras.utils import get_file

        for name in missing:
            path = get_file(name, WEIGHTS_BASE_URL + name, cache_subdir="models")
            print(f"  [ok] {name} -> {path}")
        return True
    except Exception as exc:  # noqa: BLE001 - report any network failure clearly
        print(f"\n  [ERROR] Could not download the pretrained weights: {exc}")
        print(
            "\n  Manual fix - download this file in a browser:\n"
            f"    {WEIGHTS_BASE_URL}{missing[0]}\n"
            f"  and save it as:\n    {WEIGHTS_DIR / missing[0]}\n"
            "  Then re-run this script."
        )
        return False


if __name__ == "__main__":
    ok = preflight_pretrained_weights()
    if ok:
        model, base = build_baseline_model()
        model.summary()
        print("\nSummary dict:", describe_model(model, base))
