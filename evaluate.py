import json
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from config import BATCH_SIZE, CLASS_NAMES, RESULTS_DIR
from plots import plot_confusion

INFERENCE_REPEATS = 5


def measure_inference(model, X):
    model.predict(X[:BATCH_SIZE], batch_size=BATCH_SIZE, verbose=0)
    times = []
    for _ in range(INFERENCE_REPEATS):
        start = time.perf_counter()
        model.predict(X, batch_size=BATCH_SIZE, verbose=0)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def evaluate(name, model, X_test, y_test):
    model_folder = RESULTS_DIR / name
    model_folder.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).argmax(axis=1)

    labels = list(range(len(CLASS_NAMES)))
    per_class_f1 = f1_score(y_test, y_pred, labels=labels, average=None)
    worst = int(np.argmin(per_class_f1))

    report = classification_report(y_test, y_pred, labels=labels, target_names=list(CLASS_NAMES), digits=3)
    (model_folder / "classification_report.txt").write_text(report)

    plot_confusion(y_true=y_test, y_pred=y_pred, title=f"{name}: test set", path=model_folder / "confusion.png")

    inference_s = measure_inference(model, X_test)

    test_metrics = {
        "name": name,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, labels=labels, average="macro")),
        "worst_class": str(CLASS_NAMES[worst]),
        "worst_class_f1": float(per_class_f1[worst]),
        "inference_test_set_s": round(inference_s, 3),
        "inference_ms_per_image": round(inference_s / len(X_test) * 1000, 4),
    }

    with open(model_folder / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(report)
    print(test_metrics)
    return test_metrics
