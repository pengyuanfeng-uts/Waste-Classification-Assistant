
"""
Week 1 - step 3 (run): train the frozen-base baseline model.

Trains only the classification head of MobileNetV2 on the TrashNet training split,
validates on the validation split, and writes every artefact the week-1 deliverable
needs into ``outputs/``:

    baseline_model.keras          best checkpoint (lowest val_loss)
    history_baseline.json         epoch-by-epoch metrics
    curves_baseline.png           accuracy / loss curves
    split.json                    the exact train/val/test split (shared with Week 3)
    model_summary.txt             architecture + parameter counts
    training_log.csv              raw Keras CSV log

Usage
-----
    python train_baseline.py                 # full Week 1 run (10 epochs)
    python train_baseline.py --epochs 5
    python train_baseline.py --smoke         # ~30 s sanity check, 1 epoch, 8 imgs/class
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import config
import data_pipeline as dp
import model_baseline as mb
from utils import (configure_matplotlib, ensure_dirs, header, markdown_table,
                   save_json, silence_tensorflow, step, write_text)

silence_tensorflow()
configure_matplotlib()  # no-op inside Jupyter, Agg on the command line

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

MODEL_PATH = config.OUTPUT_DIR / "baseline_model.keras"
HISTORY_PATH = config.OUTPUT_DIR / "history_baseline.json"
CURVES_PATH = config.OUTPUT_DIR / "curves_baseline.png"
CSV_LOG_PATH = config.OUTPUT_DIR / "training_log.csv"
SUMMARY_PATH = config.OUTPUT_DIR / "model_summary.txt"
SPLIT_PATH = config.OUTPUT_DIR / "split.json"


def set_reproducible(seed: int = config.SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print(f"  random seed: {seed}")


class EpochTimer(keras.callbacks.Callback):
    """Print how long each epoch takes - useful when tuning on CPU."""

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(
            f"  -> epoch {epoch + 1} took {time.time() - self._t0:.1f}s "
            f"(val_loss={logs.get('val_loss', float('nan')):.4f}, "
            f"val_accuracy={logs.get('val_accuracy', float('nan')):.4f})",
            flush=True,
        )


def build_callbacks(patience: int = 3) -> list[keras.callbacks.Callback]:
    return [
        EpochTimer(),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(CSV_LOG_PATH), append=False),
    ]


def plot_history(history: dict, path: Path) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history["accuracy"], "o-", label="train")
    axes[0].plot(epochs, history["val_accuracy"], "s-", label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["loss"], "o-", label="train")
    axes[1].plot(epochs, history["val_loss"], "s-", label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Week 1 baseline - frozen MobileNetV2 + trained classification head")
    fig.tight_layout()
    ensure_dirs(path.parent)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  curves      -> {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Week 1 baseline model.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny 1-epoch run to verify the pipeline (~1 min)")
    parser.add_argument("--rebuild-split", action="store_true",
                        help="recompute the stratified split instead of reusing split.json")
    parser.add_argument("--cache", action="store_true",
                        help="cache decoded images in RAM (faster epochs, ~0.5 GB)")
    args = parser.parse_args(argv)

    header("STEP 2/3 - train baseline (frozen MobileNetV2)", "-")
    ensure_dirs(config.OUTPUT_DIR)

    step("Environment")
    print(f"  tensorflow  : {tf.__version__}")
    print(f"  devices     : {[d.device_type for d in tf.config.list_physical_devices()]}")
    print(f"  image size  : {config.IMG_SIZE[0]}x{config.IMG_SIZE[1]}")
    print(f"  batch size  : {args.batch_size}")
    print(f"  epochs      : {args.epochs}{'  [SMOKE TEST]' if args.smoke else ''}")
    set_reproducible()

    if not mb.preflight_pretrained_weights():
        return 2

    step("Dataset split")
    splits = dp.get_or_create_split(SPLIT_PATH, rebuild=args.rebuild_split)
    print(f"  split file  : {SPLIT_PATH}")
    print(f"  seed        : {config.SEED}")
    print()
    print(dp_summary_table(splits))

    if args.smoke:
        splits = {
            name: dp.limit_per_class(items, 8) for name, items in splits.items()
        }
        args.epochs = 1
        print("\n  [SMOKE] truncated to 8 images per class per split")

    step("Building tf.data pipelines")
    datasets = {
        name: dp.build_dataset(
            items, training=(name == "train"), batch_size=args.batch_size, cache=args.cache
        )
        for name, items in splits.items()
    }
    for name, ds in datasets.items():
        print(f"  {name:<6} {len(splits[name]):>5} images, {len(ds):>3} batches")

    step("Building model")
    model, base_model = mb.build_baseline_model(learning_rate=args.learning_rate)
    info = mb.describe_model(model, base_model)
    lines = []
    model.summary(print_fn=lines.append)
    summary_text = "\n".join(lines)
    write_text(SUMMARY_PATH, summary_text + "\n")
    print(summary_text)
    print(
        f"\n  trainable parameters: {info['trainable_params']:,} / "
        f"{info['total_params']:,} total ({info['trainable_params'] / info['total_params']:.2%})"
    )
    print(f"  architecture dump -> {SUMMARY_PATH}")

    step("Training")
    t0 = time.time()
    history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=args.epochs,
        callbacks=build_callbacks(patience=3 if not args.smoke else 0),
        verbose=2,
    )
    elapsed = time.time() - t0

    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    save_json(HISTORY_PATH, {
        "config": {
            "img_size": list(config.IMG_SIZE),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs_requested": args.epochs,
            "epochs_run": len(hist["loss"]),
            "dropout": config.DROPOUT,
            "seed": config.SEED,
            "smoke_test": bool(args.smoke),
            "base_trainable": False,
        },
        "history": hist,
        "training_seconds": round(elapsed, 1),
    })
    plot_history(hist, CURVES_PATH)

    best_val_acc = max(hist["val_accuracy"])
    best_epoch = hist["val_accuracy"].index(best_val_acc) + 1
    header("Training finished")
    print(f"  epochs run          : {len(hist['loss'])}")
    print(f"  wall-clock          : {elapsed / 60:.1f} min")
    print(f"  best val_accuracy   : {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"  final train acc     : {hist['accuracy'][-1]:.4f}")
    print(f"  final val loss      : {hist['val_loss'][-1]:.4f}")
    print(f"  checkpoint          : {MODEL_PATH}")
    if args.smoke:
        print("\n  [SMOKE] Model quality is meaningless here - the point is that the")
        print("  pipeline ran end to end. Re-run without --smoke for the real baseline.")
    return 0


def dp_summary_table(splits: dict) -> str:
    summary = dp.summarise_split(splits)
    rows = []
    for name in ("train", "val", "test"):
        s = summary[name]
        rows.append([name, s["total"], *[s["per_class"][c] for c in config.CLASS_NAMES]])
    return markdown_table(["split", "total", *config.CLASS_NAMES], rows)


if __name__ == "__main__":
    raise SystemExit(main())
