from pathlib import Path

import numpy as np

SEED = 2026

IMG_SIZE = 96

TRAIN_PER_CLASS = 500
VALID_PER_CLASS = 100

BATCH_SIZE = 128

EPOCHS = 50

LEARNING_RATE = 1e-3

FT_LEARNING_RATE = 1e-5

FT_LAST_LAYERS = 20

CLASS_NAMES = np.array([
    "airplane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
])

RESULTS_DIR = Path("./results")

WEIGHT_DECAY = 1e-4
ES_PATIENCE = 8

LR_PATIENCE = 3

MIN_LR = 1e-6
