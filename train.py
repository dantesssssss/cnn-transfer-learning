import json
import time

import keras
import numpy as np
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import AdamW

from config import SEED, ES_PATIENCE, WEIGHT_DECAY, LR_PATIENCE, MIN_LR, RESULTS_DIR, EPOCHS, BATCH_SIZE
from plots import plot_history


def reset():
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)


def count_parameters(model):
    sum_trainable = 0
    for w in model.trainable_weights:
        sum_trainable += np.prod(w.shape)
    return model.count_params(), sum_trainable


def train(name, model, X_train, y_train, X_val, y_val, lr, previous_time_s=0):
    model_folder = RESULTS_DIR / name
    model_folder.mkdir(parents=True, exist_ok=True)
    model.compile(optimizer=AdamW(learning_rate=lr, weight_decay=WEIGHT_DECAY),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=ES_PATIENCE, min_delta=0.001, restore_best_weights=True)
    callbacks = [
        early_stop,
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=LR_PATIENCE, min_delta=0.001, min_lr=MIN_LR),
        CSVLogger(model_folder / "history.csv"),
    ]

    before = time.perf_counter()

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks,
                        validation_data=(X_val, y_val))
    h = history.history

    after = time.perf_counter()

    train_time = after - before
    total_params, trainable_params = count_parameters(model)

    train_loss, train_acc = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0)

    epochs_run = len(h["loss"])
    best = early_stop.best_epoch

    metrics = {
        "name": name,
        "epochs_run": epochs_run,
        "best_epoch": best + 1,
        "val_loss": float(val_loss),
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "gap": float(train_acc - val_acc),
        "loss_gap": float(val_loss - train_loss),
        "train_accuracy_during_fit": float(h["accuracy"][best]),
        "train_time_s": round(train_time, 2),
        "time_per_epoch_s": round(train_time / epochs_run, 2),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "total_time_s": round(train_time + previous_time_s, 2)
    }

    with open(model_folder / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_history(history, name, model_folder / "history.png")

    return history, metrics
