
"""
Week 1 - step 2: the data pipeline.

Builds a reproducible, stratified 70/15/15 train/validation/test split from the
TrashNet folder structure and wraps each split in an efficient ``tf.data``
pipeline (parallel decode, shuffling, prefetching).

"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import tensorflow as tf

import config
from utils import ensure_dirs, load_json, save_json, silence_tensorflow

silence_tensorflow()

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# (path as str, label as int)
Item = tuple[str, int]


# --------------------------------------------------------------------------- scanning
def scan_dataset(dataset_dir: Path | None = None) -> list[Item]:
    """Walk ``DATASET_DIR/<class>/*`` and return (relative path, label) pairs."""
    dataset_dir = Path(dataset_dir or config.DATASET_DIR)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}\n"
            "Run `python download_dataset.py` first."
        )

    items: list[Item] = []
    for label, name in enumerate(config.CLASS_NAMES):
        folder = dataset_dir / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(f"No images found in {folder}")
        # Store paths relative to DATA_DIR so the split file stays portable.
        items.extend((str(p.relative_to(config.DATA_DIR)).replace("\\", "/"), label)
                     for p in files)
    return items


# --------------------------------------------------------------------------- splitting
def stratified_split(
    items: Sequence[Item],
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    seed: int = config.SEED,
) -> tuple[list[Item], list[Item], list[Item]]:
    """Split per class so every class keeps the same ratio in all three splits."""
    rng = random.Random(seed)
    by_class: dict[int, list[Item]] = {}
    for item in items:
        by_class.setdefault(item[1], []).append(item)

    train: list[Item] = []
    val: list[Item] = []
    test: list[Item] = []

    for label in sorted(by_class):
        group = sorted(by_class[label])
        rng.shuffle(group)
        n = len(group)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        # Guarantee at least one image per split for every class.
        if n >= 5:
            n_test = max(1, min(n_test, n - 3))
            n_val = max(1, min(n_val, n - n_test - 2))
        test.extend(group[:n_test])
        val.extend(group[n_test:n_test + n_val])
        train.extend(group[n_test + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def limit_per_class(items: Sequence[Item], max_per_class: int) -> list[Item]:
    """Truncate a split so the smoke test runs in seconds."""
    seen: dict[int, int] = {}
    out: list[Item] = []
    for path, label in items:
        if seen.get(label, 0) < max_per_class:
            out.append((path, label))
            seen[label] = seen.get(label, 0) + 1
    return out


def summarise_split(splits: dict[str, Sequence[Item]]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for name, items in splits.items():
        per_class = {c: 0 for c in config.CLASS_NAMES}
        for _, label in items:
            per_class[config.CLASS_NAMES[label]] += 1
        summary[name] = {"total": len(items), "per_class": per_class}
    return summary


def split_counts(items: Iterable[Item]) -> list[int]:
    counts = [0] * config.NUM_CLASSES
    for _, label in items:
        counts[label] += 1
    return counts


# --------------------------------------------------------------------------- persistence
def save_split(path: Path, splits: dict[str, Sequence[Item]]) -> None:
    payload = {
        "seed": config.SEED,
        "val_ratio": config.VAL_RATIO,
        "test_ratio": config.TEST_RATIO,
        "class_names": config.CLASS_NAMES,
        "splits": {name: [list(pair) for pair in items] for name, items in splits.items()},
    }
    save_json(path, payload)


def load_split(path: Path) -> dict[str, list[Item]]:
    payload = load_json(path)
    return {
        name: [(str(p), int(y)) for p, y in pairs]
        for name, pairs in payload["splits"].items()
    }


def get_or_create_split(cache_path: Path | None = None,
                        rebuild: bool = False) -> dict[str, list[Item]]:
    """Load the persisted split, creating it from disk on first run."""
    cache_path = Path(cache_path or (config.OUTPUT_DIR / "split.json"))
    if cache_path.exists() and not rebuild:
        splits = load_split(cache_path)
        # Guard against a stale cache pointing at a removed dataset.
        if splits.get("train"):
            return splits

    items = scan_dataset()
    train, val, test = stratified_split(items)
    splits = {"train": train, "val": val, "test": test}
    ensure_dirs(cache_path.parent)
    save_split(cache_path, splits)
    return splits


# --------------------------------------------------------------------------- tf.data
def _decode_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(path)
    # JPEGs from TrashNet are RGB; decode eagerly so failures surface immediately.
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, config.IMG_SIZE, method="bilinear")
    # Keep 0..255 float32 - the model performs MobileNetV2 pre-processing itself.
    return tf.cast(image, tf.float32), label


def build_dataset(
    items: Sequence[Item],
    *,
    training: bool,
    batch_size: int = config.BATCH_SIZE,
    shuffle_buffer: int = 1000,
    cache: bool = config.CACHE_AFTER_RESIZE,
) -> tf.data.Dataset:
    """Turn an (path, label) list into a batched, prefetched ``tf.data.Dataset``."""
    paths = [str(config.DATA_DIR / p) for p, _ in items]
    labels = [y for _, y in items]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), shuffle_buffer), seed=config.SEED,
                        reshuffle_each_iteration=True)

    ds = ds.map(_decode_resize, num_parallel_calls=AUTOTUNE, deterministic=not training)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    if training:
        ds = ds.shuffle(min(len(paths), shuffle_buffer) // batch_size + 1,
                        seed=config.SEED)
    return ds.prefetch(AUTOTUNE)


def build_all(
    splits: dict[str, Sequence[Item]] | None = None,
    *,
    batch_size: int = config.BATCH_SIZE,
) -> dict[str, tf.data.Dataset]:
    splits = splits or get_or_create_split()
    return {
        "train": build_dataset(splits["train"], training=True, batch_size=batch_size),
        "val": build_dataset(splits["val"], training=False, batch_size=batch_size),
        "test": build_dataset(splits["test"], training=False, batch_size=batch_size),
    }


def compute_class_weight(items: Sequence[Item]) -> dict[int, float]:
    """Balanced class weights - used from Week 2 as an imbalance experiment."""
    counts = split_counts(items)
    total = sum(counts)
    return {
        i: total / (config.NUM_CLASSES * c) if c else 0.0
        for i, c in enumerate(counts)
    }
