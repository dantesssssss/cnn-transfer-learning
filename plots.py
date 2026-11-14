import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import confusion_matrix

from config import CLASS_NAMES

TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED = "#52514e"
GRID = "#e5e4e0"
INK = "#1f1f1d"
FAINT = "#a3a29d"
BLUES = LinearSegmentedColormap.from_list(  # one hue, light -> dark (more images = darker)
    "blues", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _acc_key(h):
    return next(k for k in h if "accuracy" in k and not k.startswith("val_"))


def plot_history(history, title, path):
    h = history.history
    acc_key = _acc_key(h)
    epochs = np.arange(1, len(h["loss"]) + 1)
    best = int(np.argmin(h["val_loss"]))

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in [(ax_loss, "loss", "Loss"), (ax_acc, acc_key, "Accuracy")]:
        ax.plot(epochs, h[key], color=TRAIN_COLOR, linewidth=2, label="training")
        ax.plot(epochs, h["val_" + key], color=VAL_COLOR, linewidth=2, label="validation")
        _style(ax, "Epoch", label)
        ax.legend(frameon=False)
    ax_loss.plot(epochs[best], h["val_loss"][best], "o", color=VAL_COLOR,
                 markersize=8, markeredgecolor="white", markeredgewidth=2)
    ax_loss.set_title(f"lowest validation loss: epoch {best + 1} (dot)", fontsize=10, color=MUTED)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Lowest validation loss {h['val_loss'][best]:.4f} at epoch {best + 1}; saved: {path}")


def plot_compare(*, histories, title, path):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.6))
    rows = []
    for (name, history), color in zip(histories.items(), RUN_COLORS):
        h = history.history
        acc_key = _acc_key(h)
        epochs = np.arange(1, len(h["loss"]) + 1)
        for ax, key in [(ax_loss, "loss"), (ax_acc, acc_key)]:
            ax.plot(epochs, h["val_" + key], color=color, linewidth=2)
            ax.plot(epochs, h[key], color=color, linewidth=1.5, linestyle="--")
        best = int(np.argmin(h["val_loss"]))
        rows.append({
            "run": name,
            "epochs": len(epochs),
            "best_val_loss": round(h["val_loss"][best], 4),
            "best_epoch": best + 1,
            "val_acc_at_best": round(h["val_" + acc_key][best], 4),
            "train_acc_at_best": round(h[acc_key][best], 4),
            "val_loss_last": round(h["val_loss"][-1], 4),
            "train_acc_last": round(h[acc_key][-1], 4),
            "val_acc_last": round(h["val_" + acc_key][-1], 4),
        })
    _style(ax_loss, "Epoch", "Loss")
    _style(ax_acc, "Epoch", "Accuracy")

    handles = [Line2D([], [], color=c, linewidth=2, label=n) for n, c in zip(histories, RUN_COLORS)]
    handles += [Line2D([], [], color=MUTED, linewidth=2, label="validation"),
                Line2D([], [], color=MUTED, linewidth=1.5, linestyle="--", label="training")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)

    with open(Path(path).with_suffix(".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{title}")
    print(f"  {'run':<30} {'epochs':>6} {'best val loss':>13} {'@epoch':>6} {'val acc':>8} {'train acc':>9}"
          f" {'last val loss':>13}")
    for r in rows:
        print(f"  {r['run']:<30} {r['epochs']:>6} {r['best_val_loss']:>13.4f} {r['best_epoch']:>6}"
              f" {r['val_acc_at_best']:>8.4f} {r['train_acc_at_best']:>9.4f} {r['val_loss_last']:>13.4f}")
    print("  (val acc / train acc: at the best epoch)  saved:", path)


def plot_confusion(*, y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)  # rows: true class; columns: predicted class
    n = len(CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap=BLUES, vmin=0, vmax=cm.max())
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            color = "white" if v > 0.55 * cm.max() else (INK if v else FAINT)
            ax.text(j, i, v, ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(n), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(n), CLASS_NAMES)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)  # white gaps between cells
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("diagonal = correct; everything else = a mistake", fontsize=10, color=MUTED)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    mistakes = [(cm[i, j], i, j) for i in range(n) for j in range(n) if i != j]
    print("Most common mistakes (true -> predicted):")
    for count, i, j in sorted(mistakes, reverse=True)[:5]:
        print(f"  {CLASS_NAMES[i]:>10} -> {CLASS_NAMES[j]:<10} {count}")
    print("saved:", path)


MODEL_LABELS = {"cnn": "Own CNN", "fe": "Feature extraction", "ft": "Fine-tuning"}
AUGMENT_LABELS = {False: "no augmentation", True: "augmentation"}


def _compact(value):
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{value:.0f}"


def plot_experiment(*, rows, path):
    panels = [
        ("test_accuracy", "Test accuracy (%)", 100, "{:.1f}", False),
        ("gap", "Gap: train − validation accuracy (pp)", 100, "{:.1f}", False),
        ("total_time_s", "Training time, total (s)", 1, "{:.0f}", False),
        ("trainable_params", "Trainable parameters (log scale)", 1, None, True),
    ]
    lookup = {(r["model"], r["augment"]): r for r in rows}
    models = list(MODEL_LABELS)
    x = np.arange(len(models))
    width = 0.3

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    for ax, (key, label, scale, fmt, log) in zip(axes.flat, panels):
        for k, augment in enumerate((False, True)):
            values = [scale * float(lookup[(m, augment)][key]) for m in models]
            offset = (k - 0.5) * (width + 0.04)
            bars = ax.bar(x + offset, values, width, color=RUN_COLORS[k])
            for bar, value in zip(bars, values):
                text = _compact(value) if log else fmt.format(value)
                ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, color=MUTED)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(1e3, 1e7)
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}K" if v >= 1e3 else f"{v:g}"))
        else:
            ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
        ax.axhline(0, color=FAINT, linewidth=1)
        ax.set_xticks(x, [MODEL_LABELS[m] for m in models])
        ax.tick_params(axis="x", length=0)
        _style(ax, "", label)
        ax.grid(axis="x", visible=False)

    handles = [Patch(color=RUN_COLORS[k], label=AUGMENT_LABELS[a]) for k, a in enumerate((False, True))]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Own CNN vs feature extraction vs fine-tuning, with and without augmentation")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved:", path)
