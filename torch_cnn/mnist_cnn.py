"""Small PyTorch CNN on MNIST, with the measured part wrapped in ROI markers.

Model: conv(1->8,3x3) relu maxpool2 -> conv(8->16,3x3) relu maxpool2 -> fc(400->10)

usage: python3 mnist_cnn.py --mode infer|train [--steps N] [--batch B]
                            [--seed S] [--offset K] [--warmup W]
Only the --steps measured iterations sit inside the region of interest;
import, data loading, model construction and --warmup iterations do not.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import roi

MNIST = Path(__file__).resolve().parent.parent / "CNN/AxelMontlahuc/MNIST"


def read_idx(path):
    data = path.read_bytes()
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load(split, offset, count):
    prefix = "train" if split == "train" else "t10k"
    images = read_idx(MNIST / f"{prefix}-images.idx3-ubyte")[offset:offset + count]
    labels = read_idx(MNIST / f"{prefix}-labels.idx1-ubyte")[offset:offset + count]
    x = torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)
    return x, torch.from_numpy(labels.astype(np.int64))


def make_model():
    return nn.Sequential(
        nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 5 * 5, 10),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["infer", "train"], default="infer")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    n = (args.warmup + args.steps) * args.batch
    x, y = load("train" if args.mode == "train" else "test", args.offset, n)
    model = make_model()
    batches = [(x[i:i + args.batch], y[i:i + args.batch]) for i in range(0, n, args.batch)]

    if args.mode == "train":
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        def step(xb, yb):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            return loss.item()
    else:
        model.eval()

        @torch.no_grad()
        def step(xb, yb):
            return (model(xb).argmax(1) == yb).float().mean().item()

    for xb, yb in batches[:args.warmup]:
        step(xb, yb)

    t0 = time.perf_counter()
    roi.begin()
    results = [step(xb, yb) for xb, yb in batches[args.warmup:]]
    roi.end()
    dt = time.perf_counter() - t0

    what = "loss" if args.mode == "train" else "accuracy"
    print(f"{args.mode}: {args.steps} x batch {args.batch}, mean {what} "
          f"{sum(results) / len(results):.4f}, {dt:.2f}s in ROI")


if __name__ == "__main__":
    main()
