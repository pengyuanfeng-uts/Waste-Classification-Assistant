
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
# TrashNet (Yang & Thung, 2016): 6 classes, ~2 527 RGB photos at 512x384.
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASS_NAMES)

# --------------------------------------------------------------------------- split
# Reproducible, stratified 70 / 15 / 15 split. The test split is written to disk
# so that Week 3 evaluates on EXACTLY the same held-out images.
SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15                                # -> train = 0.70

# --------------------------------------------------------------------------- pipeline
# MobileNetV2 ships ImageNet weights for exactly 96/128/160/192/224. The plan sent
# to the tutor used 150, but Keras silently swaps 150 for the 224 weights file and
# logs a warning; 160 is natively supported, is within ~14% of 150 in pixel count,
# and keeps CPU training fast. Week 3 revisits 224 as an ablation.
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
# Decoding + resizing on the fly costs a little CPU per epoch but keeps RAM low.
# Set to True only if the machine has >= 8 GB free (the resized train split alone
# is roughly 480 MB as float32).
CACHE_AFTER_RESIZE = False

# --------------------------------------------------------------------------- baseline model
LEARNING_RATE = 1e-3
DROPOUT = 0.2
EPOCHS = 10
# TrashNet is imbalanced (trash has ~137 images vs. paper's ~594). Turning class
# weights on is a Week 2 experiment, so it stays off in the baseline.
USE_CLASS_WEIGHT = False

# --------------------------------------------------------------------------- evaluation
# Minimum fraction of "percentage of examples correctly classified" below which a
# class is flagged as weak in the generated report.
WEAK_CLASS_RECALL_THRESHOLD = 0.60

# --------------------------------------------------------------------------- dataset source
# Tried in order. The canonical GitHub URL is the slow-but-reliable fallback.
DATASET_URLS = [
    "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip",
    "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip",
]
DATASET_EXPECTED_BYTES = 42_834_870
