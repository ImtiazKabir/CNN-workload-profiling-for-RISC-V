"""Convert MNIST IDX files into the raw float32 .mat files magicalbat expects.

Same output as CNN/magicalbat/mnist.py (images /255, labels as float32),
without needing tensorflow.
usage: make_mat.py <mnist_idx_dir> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np


def read_idx(path):
    data = path.read_bytes()
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for split, prefix in (("train", "train"), ("test", "t10k")):
        images = read_idx(src / f"{prefix}-images.idx3-ubyte").astype(np.float32) / 255.0
        labels = read_idx(src / f"{prefix}-labels.idx1-ubyte").astype(np.float32)
        images.tofile(out / f"{split}_images.mat")
        labels.tofile(out / f"{split}_labels.mat")
        print(split, images.shape, labels.shape)


if __name__ == "__main__":
    main()
