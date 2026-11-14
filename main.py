import numpy as np

from config import LEARNING_RATE
from data import load_splits
from models import build_cnn, build_transfer
from train import reset, train

# data
X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

# CNN from scratch
reset()

model = build_cnn(augment=False)

_, metrics = train("cnn_noaug", model, X_train, y_train, X_val, y_val, LEARNING_RATE)

print(metrics)

# Frozen backbone of mobilenet_v2
reset()
model, backbone = build_transfer(augment=False)
bef = backbone.weights[0].numpy().copy()
train("fe_noaug", model, X_train, y_train, X_val, y_val, LEARNING_RATE)
aft = backbone.weights[0].numpy().copy()

print(f"Is backbone frozen: {np.array_equal(bef, aft)}")
