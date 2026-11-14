import numpy as np

from config import LEARNING_RATE, FT_LAST_LAYERS, FT_LEARNING_RATE
from data import load_splits
from evaluate import evaluate
from models import build_cnn, build_transfer, enable_fine_tuning
from train import reset, train

# data
X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

# CNN from scratch
reset()
model = build_cnn(augment=False)
_, metrics = train("cnn_noaug", model, X_train, y_train, X_val, y_val, LEARNING_RATE)
print(metrics)
evaluate("cnn_noaug", model, X_test, y_test)

# Frozen backbone of mobilenet_v2
reset()
model, backbone = build_transfer(augment=False)
first_before = backbone.weights[0].numpy().copy()
_, metrics = train("fe_noaug", model, X_train, y_train, X_val, y_val, LEARNING_RATE)
print(metrics)
print(f"FE - first layer unchanged (frozen): {np.array_equal(first_before, backbone.weights[0].numpy())}")
evaluate("fe_noaug", model, X_test, y_test)

# Unfreezing some layers for backbone of mobilenet_v2
print(f"Unfreezing the last {FT_LAST_LAYERS} layers of the backbone")
enable_fine_tuning(backbone, FT_LAST_LAYERS)
first_before = backbone.weights[0].numpy().copy()
last_before = backbone.get_layer("Conv_1").weights[0].numpy().copy()
_, ft_metrics = train("ft_noaug", model, X_train, y_train, X_val, y_val, FT_LEARNING_RATE,
                      previous_time_s=metrics["total_time_s"])
print(ft_metrics)
print(f"FT - first layer unchanged (frozen): {np.array_equal(first_before, backbone.weights[0].numpy())}")
print(f"FT - Conv_1 changed (fine-tuned): "
      f"{not np.array_equal(last_before, backbone.get_layer('Conv_1').weights[0].numpy())}")
evaluate("ft_noaug", model, X_test, y_test)
