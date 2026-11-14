import numpy as np

from config import LEARNING_RATE, FT_LAST_LAYERS, FT_LEARNING_RATE
from data import load_splits
from models import build_cnn, build_transfer, enable_fine_tuning
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
history, metrics = train("fe_noaug", model, X_train, y_train, X_val, y_val, LEARNING_RATE)
aft = backbone.weights[0].numpy().copy()

print(f"Is backbone frozen: {np.array_equal(bef, aft)}")

# Unfreezing some layers for backbone of mobilenet_v2
print(f"now unfreezing last {FT_LAST_LAYERS} layers of backbone")
enable_fine_tuning(backbone, FT_LAST_LAYERS)
bef = backbone.get_layer("Conv_1").weights[0].numpy().copy()
train("ft_noaug", model, X_train, y_train, X_val, y_val, FT_LEARNING_RATE, previous_time_s=metrics["total_time_s"])
aft = backbone.get_layer("Conv_1").weights[0].numpy().copy()
print(f"Are last {FT_LAST_LAYERS} layers of backbone are not frozen: {not np.array_equal(bef, aft)}")