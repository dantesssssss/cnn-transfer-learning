import csv

import numpy as np

from config import LEARNING_RATE, FT_LAST_LAYERS, FT_LEARNING_RATE, RESULTS_DIR
from data import load_splits
from evaluate import evaluate
from models import build_cnn, build_transfer, enable_fine_tuning
from plots import plot_compare, plot_experiment
from train import reset, train

COLUMNS = [
    "name", "model", "augment",
    "test_accuracy", "test_macro_f1", "worst_class", "worst_class_f1",
    "val_accuracy", "val_loss", "train_accuracy", "gap", "loss_gap",
    "epochs_run", "best_epoch", "train_time_s", "total_time_s", "time_per_epoch_s",
    "inference_ms_per_image", "trainable_params", "total_params",
]


def collect(model_type, augment, metrics, test_metrics):
    return {**metrics, **test_metrics, "model": model_type, "augment": augment}


def write_results(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("saved:", path)


def first_layer(backbone):
    return backbone.weights[0].numpy().copy()


def last_conv(backbone):
    return backbone.get_layer("Conv_1").weights[0].numpy().copy()


X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

rows = []
histories = {}

for augment in (False, True):
    tag = "aug" if augment else "noaug"

    name = f"cnn_{tag}"
    reset()
    model = build_cnn(augment=augment)
    histories[name], metrics = train(name, model, X_train, y_train, X_val, y_val, LEARNING_RATE)
    rows.append(collect("cnn", augment, metrics, evaluate(name, model, X_test, y_test)))

    name = f"fe_{tag}"
    reset()
    model, backbone = build_transfer(augment=augment)
    first_before = first_layer(backbone)
    histories[name], fe_metrics = train(name, model, X_train, y_train, X_val, y_val, LEARNING_RATE)
    print(f"{name} - first layer unchanged (frozen): {np.array_equal(first_before, first_layer(backbone))}")
    rows.append(collect("fe", augment, fe_metrics, evaluate(name, model, X_test, y_test)))

    name = f"ft_{tag}"
    print(f"Unfreezing the last {FT_LAST_LAYERS} layers of the backbone")
    enable_fine_tuning(backbone, FT_LAST_LAYERS)
    first_before, last_before = first_layer(backbone), last_conv(backbone)
    histories[name], ft_metrics = train(name, model, X_train, y_train, X_val, y_val, FT_LEARNING_RATE,
                                        previous_time_s=fe_metrics["total_time_s"])
    print(f"{name} - first layer unchanged (frozen): {np.array_equal(first_before, first_layer(backbone))}")
    print(f"{name} - Conv_1 changed (fine-tuned): {not np.array_equal(last_before, last_conv(backbone))}")
    rows.append(collect("ft", augment, ft_metrics, evaluate(name, model, X_test, y_test)))

write_results(rows, RESULTS_DIR / "results.csv")
plot_experiment(rows=rows, path=RESULTS_DIR / "experiment.png")
for model_type in ("cnn", "fe", "ft"):
    plot_compare(
        histories={f"{model_type}_noaug": histories[f"{model_type}_noaug"],
                   f"{model_type}_aug": histories[f"{model_type}_aug"]},
        title=f"{model_type.upper()}: without vs with augmentation",
        path=RESULTS_DIR / f"compare_{model_type}.png",
    )
