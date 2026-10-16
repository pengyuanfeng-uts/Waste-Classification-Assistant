
"""
Week 1 - step 4: evaluation and error analysis.

Loads the trained baseline and produces the multi-dimensional evaluation the
project brief asks for:

* overall accuracy **and** macro / weighted F1 (accuracy alone hides the fact that
  TrashNet is imbalanced)
* confusion matrix, raw counts and row-normalised
* per-class precision / recall / F1 / support
* the most frequently confused class pairs
* a gallery of misclassified images, so the errors can actually be looked at

Artefacts written to ``outputs/``:
    metrics_<split>.json                numeric results
    classification_report_<split>.txt   sklearn text report
    confusion_matrix_<split>.png        counts + normalised heat maps
    misclassified_<split>.png           grid of wrong predictions
    week1_report.md                     ready-to-paste summary

Usage
-----
    python evaluate.py                       # evaluate the saved baseline
    python evaluate.py --split test
    python evaluate.py --model outputs/baseline_model.keras

References (APA 7)
------------------
Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in
    Science & Engineering, 9(3), 90-95. https://doi.org/10.1109/MCSE.2007.55
    -> every figure written by this module.
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel,
    O., ... Duchesnay, E. (2011). Scikit-learn: Machine learning in Python.
    Journal of Machine Learning Research, 12, 2825-2830.
    -> confusion_matrix, classification_report, precision/recall/F1 and
       balanced accuracy; the choice of macro-F1 over raw accuracy is
       recommended there for imbalanced targets.
Yang, M., & Thung, G. (2016). Classification of trash for recyclability
    status (CS229 Project Report). Stanford University.
    -> the baseline numbers this evaluation is compared against.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import config
import data_pipeline as dp
from utils import (configure_matplotlib, ensure_dirs, header, markdown_table,
                   save_json, silence_tensorflow, step, write_text)

silence_tensorflow()
configure_matplotlib()  # no-op inside Jupyter, Agg on the command line

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)

SPLIT_PATH = config.OUTPUT_DIR / "split.json"
DEFAULT_MODEL = config.OUTPUT_DIR / "baseline_model.keras"
SHORT = [c[:4] for c in config.CLASS_NAMES]

# (glob pattern, description) for every artefact this project can write. The report
# only lists the ones that are actually present, so the same table is correct whether
# the project is driven from the notebook or from the command-line scripts.
OUTPUT_ARTEFACTS = [
    ("baseline_model.keras", "best checkpoint by validation loss"),
    ("training_log.csv", "raw per-epoch Keras CSV log"),
    ("history_baseline.json", "per-epoch loss / accuracy (written by train_baseline.py)"),
    ("curves_baseline.png", "accuracy and loss curves"),
    ("confusion_matrix_*.png", "confusion matrices (counts + row-normalised)"),
    ("classification_report_*.txt", "sklearn per-class report"),
    ("metrics_*.json", "all numeric results, including raw predictions"),
    ("misclassified_*.png", "gallery of confident mistakes"),
    ("model_summary.txt", "network summary and parameter counts"),
    ("split.json", "the exact split reused in Week 3"),
    ("week1_report.md", "this report"),
]


# --------------------------------------------------------------------------- inference
def predict(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a dataset in batches; returns (y_true, y_prob)."""
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, batch_labels in dataset:
        preds = model.predict(images, verbose=0)
        probs.append(np.asarray(preds))
        labels.append(np.asarray(batch_labels).reshape(-1))
    return np.concatenate(labels), np.concatenate(probs)


# --------------------------------------------------------------------------- metrics
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = y_prob.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(config.CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_conf": y_prob.max(axis=1).tolist(),
    }


def top_confusions(cm: np.ndarray, k: int = 5) -> list[tuple[str, str, int]]:
    """Most frequent off-diagonal cells: (true class, predicted class, count)."""
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((config.CLASS_NAMES[i], config.CLASS_NAMES[j], int(cm[i, j])))
    pairs.sort(key=lambda p: -p[2])
    return pairs[:k]


def weak_classes(metrics: dict, threshold: float = config.WEAK_CLASS_RECALL_THRESHOLD) -> list[str]:
    return [
        name for name, m in metrics["per_class"].items()
        if m["recall"] < threshold
    ]


# --------------------------------------------------------------------------- plots
def plot_confusion(cm: np.ndarray, path: Path, split: str, accuracy: float,
                   label: str = "Week 1 baseline") -> None:
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    for ax, matrix, title, fmt in (
        (axes[0], cm, "Confusion matrix - counts", "d"),
        (axes[1], cm_norm, "Confusion matrix - row-normalised (recall)", ".2f"),
    ):
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() or 1)
        ax.set_title(title)
        ax.set_xticks(range(config.NUM_CLASSES))
        ax.set_yticks(range(config.NUM_CLASSES))
        ax.set_xticklabels(SHORT, rotation=45, ha="right")
        ax.set_yticklabels(SHORT)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        for i in range(config.NUM_CLASSES):
            for j in range(config.NUM_CLASSES):
                value = matrix[i, j]
                if fmt == "d":
                    text = f"{int(value)}"
                else:
                    text = f"{value:.2f}".replace("0.", ".")
                ax.text(
                    j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if value > (matrix.max() * 0.55) else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"{label} - {split} split (accuracy {accuracy:.1%})")
    fig.tight_layout()
    ensure_dirs(path.parent)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  confusion matrix -> {path}")


def plot_misclassified(
    y_true: np.ndarray, y_pred: np.ndarray, y_conf: np.ndarray,
    items: list[tuple[str, int]], path: Path, split: str, max_images: int = 16,
    label: str = "Week 1 baseline",
) -> int:
    """Save a gallery of confidently-wrong predictions (most useful to inspect)."""
    wrong = np.where(y_true != y_pred)[0]
    if wrong.size == 0:
        print("  no misclassified images to plot")
        return 0

    order = wrong[np.argsort(-y_conf[wrong])][:max_images]
    cols = 4
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, order):
        rel_path, _ = items[idx]
        image = tf.io.decode_image(
            tf.io.read_file(str(config.DATA_DIR / rel_path)), channels=3,
            expand_animations=False,
        ).numpy()
        ax.imshow(image)
        ax.set_title(
            f"true: {config.CLASS_NAMES[int(y_true[idx])]}\n"
            f"pred: {config.CLASS_NAMES[int(y_pred[idx])]} ({y_conf[idx]:.0%})",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes[len(order):]:
        ax.axis("off")

    fig.suptitle(f"{label} - most confident mistakes on the {split} split")
    fig.tight_layout()
    ensure_dirs(path.parent)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  error gallery    -> {path}")
    return int(wrong.size)


# --------------------------------------------------------------------------- report
def write_report(results: dict[str, dict], splits: dict, model_info: dict | None,
                 path: Path, model_path: Path) -> None:
    summary = dp.summarise_split(splits)
    split_rows = [
        [name, summary[name]["total"], *[summary[name]["per_class"][c] for c in config.CLASS_NAMES]]
        for name in ("train", "val", "test")
    ]

    lines: list[str] = []
    lines.append("# Week 1 - baseline deliverable report\n")
    lines.append(
        "Automatic summary generated by `week1/evaluate.py`. "
        "Numbers come straight from `outputs/metrics_*.json` - nothing is typed by hand.\n"
    )

    lines.append("## 1. Model\n")
    lines.append(f"- Checkpoint: `{model_path.name}`")
    lines.append(f"- Architecture: MobileNetV2 (ImageNet, frozen) + GlobalAveragePooling2D + Dropout + Dense(6, softmax)")
    lines.append(f"- Input: {config.IMG_SIZE[0]}x{config.IMG_SIZE[1]}, batch {config.BATCH_SIZE}, seed {config.SEED}")
    if model_info:
        lines.append(
            f"- Trainable parameters: {model_info['trainable_params']:,} of "
            f"{model_info['total_params']:,} "
            f"({model_info['trainable_params'] / model_info['total_params']:.2%})"
        )
    lines.append("")

    lines.append("## 2. Data split (stratified, reproducible)\n")
    lines.append(markdown_table(["split", "total", *config.CLASS_NAMES], split_rows))
    lines.append("")

    lines.append("## 3. Headline metrics\n")
    rows = []
    for split, metrics in results.items():
        rows.append([
            split,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['balanced_accuracy']:.4f}",
            f"{metrics['macro_f1']:.4f}",
            f"{metrics['weighted_f1']:.4f}",
        ])
    lines.append(markdown_table(["split", "accuracy", "balanced acc.", "macro F1", "weighted F1"], rows))
    lines.append("")

    for idx, (split, metrics) in enumerate(results.items()):
        lines.append(f"## 4{chr(97 + idx)}. Per-class performance on `{split}`\n")
        rows = [
            [name, f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}", m["support"]]
            for name, m in metrics["per_class"].items()
        ]
        lines.append(markdown_table(["class", "precision", "recall", "F1", "support"], rows))
        lines.append("")
        weak = weak_classes(metrics)
        if weak:
            lines.append(
                f"- Below the {config.WEAK_CLASS_RECALL_THRESHOLD:.0%} recall threshold: "
                f"**{', '.join(weak)}** - these are the targets for the Week 2 fine-tuning "
                "and class-weight experiments.\n"
            )
        else:
            lines.append(
                f"- Every class is at or above the "
                f"{config.WEAK_CLASS_RECALL_THRESHOLD:.0%} recall threshold.\n"
            )
        pairs = top_confusions(np.asarray(metrics["confusion_matrix"]))
        if pairs:
            lines.append("Most frequent confusions (actual -> predicted):\n")
            lines.append(markdown_table(
                ["actual", "predicted", "count"],
                [[a, p, n] for a, p, n in pairs],
            ))
            lines.append("")

    lines.append("## 5. Files produced\n")
    lines.append(
        "Listed by scanning `outputs/`, so this table can only ever describe files that "
        "genuinely exist - it cannot drift out of sync with what was actually written.\n"
    )
    produced = []
    for pattern, description in OUTPUT_ARTEFACTS:
        matches = sorted(config.OUTPUT_DIR.glob(pattern))
        if matches:
            produced.append([", ".join(p.name for p in matches), description])
    lines.append(markdown_table(["file", "content"], produced) if produced
                 else "_No artefacts found in `outputs/`._")
    lines.append("")

    write_text(path, "\n".join(lines))
    print(f"\n  week 1 report    -> {path}")


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Week 1 baseline model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--no-report", action="store_true",
                        help="skip regenerating week1_report.md")
    args = parser.parse_args(argv)

    header("STEP 3/3 - evaluation & error analysis", "-")

    if not Path(args.model).exists():
        print(f"[ERROR] Model not found: {args.model}")
        print("        Run `python train_baseline.py` first.")
        return 2

    step(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)
    trainable_params = int(sum(int(tf.size(w)) for w in model.trainable_weights))
    model_info = {
        "total_params": int(model.count_params()),
        "trainable_params": trainable_params,
    }

    splits = dp.get_or_create_split(SPLIT_PATH)
    names = ["val", "test"] if args.split == "both" else [args.split]
    results: dict[str, dict] = {}

    for name in names:
        step(f"Evaluating the {name} split ({len(splits[name])} images)")
        ds = dp.build_dataset(splits[name], training=False, batch_size=args.batch_size)
        y_true, y_prob = predict(model, ds)
        metrics = compute_metrics(y_true, y_prob)
        results[name] = metrics

        print(classification_report(
            y_true, y_prob.argmax(axis=1),
            labels=list(range(config.NUM_CLASSES)),
            target_names=config.CLASS_NAMES, digits=3, zero_division=0,
        ))
        print(f"  accuracy        : {metrics['accuracy']:.4f}")
        print(f"  balanced acc.   : {metrics['balanced_accuracy']:.4f}")
        print(f"  macro F1        : {metrics['macro_f1']:.4f}")

        cm = np.asarray(metrics["confusion_matrix"])
        print("  most confused pairs:")
        for actual, predicted, count in top_confusions(cm, k=3):
            print(f"    {actual:<10} -> {predicted:<10} {count:>4}x")

        write_text(
            config.OUTPUT_DIR / f"classification_report_{name}.txt",
            classification_report(
                y_true, y_prob.argmax(axis=1),
                labels=list(range(config.NUM_CLASSES)),
                target_names=config.CLASS_NAMES, digits=4, zero_division=0,
            ),
        )
        plot_confusion(cm, config.OUTPUT_DIR / f"confusion_matrix_{name}.png",
                       name, metrics["accuracy"])
        n_wrong = plot_misclassified(
            y_true, y_prob.argmax(axis=1), y_prob.max(axis=1),
            splits[name], config.OUTPUT_DIR / f"misclassified_{name}.png", name,
        )
        metrics["misclassified_count"] = n_wrong

        save_json(config.OUTPUT_DIR / f"metrics_{name}.json", metrics)

    if not args.no_report:
        write_report(results, splits, model_info,
                     config.OUTPUT_DIR / "week1_report.md", Path(args.model))

    header("Evaluation finished")
    for name, metrics in results.items():
        print(f"  {name:<5} accuracy={metrics['accuracy']:.4f}  macroF1={metrics['macro_f1']:.4f}"
              f"  errors={metrics['misclassified_count']}")
    print(f"\n  outputs -> {config.OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
