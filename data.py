import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10

from config import RESULTS_DIR, SEED, TRAIN_PER_CLASS, CLASS_NAMES, VALID_PER_CLASS


def load_splits():
    (X_full, y_full), (X_test, y_test) = cifar10.load_data()

    y_full = np.array(y_full).ravel()
    y_test = np.array(y_test).ravel()

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, train_size=TRAIN_PER_CLASS * CLASS_NAMES.size, test_size=VALID_PER_CLASS * CLASS_NAMES.size,
        stratify=y_full, random_state=SEED,
    )

    print("Training / validation / test:", len(X_train), len(X_val), len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


def show_samples(x, y, path):
    if not (path).exists():
        fig, axes = plt.subplots(2, 5, figsize=(10, 5.2))
        for c, ax in enumerate(axes.flat):
            i = np.flatnonzero(y == c)[0]
            ax.imshow(x[i])
            ax.set_title(CLASS_NAMES[c], fontsize=9)
            ax.axis("off")
        fig.suptitle("CIFAR-10: one training image per class")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

    print(np.bincount(y_train), np.bincount(y_val), np.bincount(y_test))

    print(X_train.dtype, X_train.min(), X_train.max())
    show_samples(X_train, y_train, RESULTS_DIR / "samples.png")
