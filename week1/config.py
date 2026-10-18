
"""
Week 1 - shared configuration.

Single source of truth for paths, class names, the train/val/test split and the
baseline hyper-parameters. Every other module in this folder imports from here,
so a value only ever has to be changed in one place.
"""

from pathlib import Path

# --------------------------------------------------------------------------- paths
WEEK_DIR = Path(__file__).resolve().parent
DATA_DIR = WEEK_DIR / "data"
ZIP_PATH = DATA_DIR / "dataset-resized.zip"
DATASET_DIR = DATA_DIR / "dataset-resized"      # DATASET_DIR/<class_name>/*.jpg
OUTPUT_DIR = WEEK_DIR / "outputs"

# --------------------------------------------------------------------------- dataset

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASS_NAMES)

# --------------------------------------------------------------------------- split
# Reproducible, stratified 70 / 15 / 15 split. The test split is written to disk
# so that Week 3 evaluates on EXACTLY the same held-out images.
SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15                                # -> train = 0.70


IMG_SIZE = (160, 160)
BATCH_SIZE = 32

CACHE_AFTER_RESIZE = False

# --------------------------------------------------------------------------- baseline model
LEARNING_RATE = 1e-3
DROPOUT = 0.2
EPOCHS = 10

USE_CLASS_WEIGHT = False

WEAK_CLASS_RECALL_THRESHOLD = 0.60

DATASET_URLS = [
    "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip",
    "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip",
]
DATASET_EXPECTED_BYTES = 42_834_870
