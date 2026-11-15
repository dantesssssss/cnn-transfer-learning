# CNN and transfer learning

Own CNN vs. MobileNetV2 (feature extraction and fine-tuning) on CIFAR-10, with and without data augmentation.

[Full report (PDF)](CNN_TL.pdf)

## Run

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The pinned `tensorflow-metal` uses the Apple Silicon GPU.

```bash
uv sync
uv run python main.py
```

`main.py` trains all six variants and writes metrics, plots and `results.csv` to `results/`. CIFAR-10 and the MobileNetV2 weights are downloaded on the first run.

To only check the data splits (prints their shapes and class counts, and saves `results/samples.png` if it doesn't exist yet):

```bash
uv run python data.py
```
