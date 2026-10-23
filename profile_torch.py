"""Dynamic RISC-V opcode counts of a PyTorch MNIST CNN (torch_cnn/mnist_cnn.py).

Runs CPython + Debian's riscv64 torch from third_party/rv-rootfs under QEMU with
tools/opcount_roi.so. Counts cover the whole process ("all") and the region of
interest ("roi" = the measured train/infer steps), split by shared object.

Configurations: mode {infer, train} x BLAS {openblas, reference}
                x cpu {max (RVV), max,v=false}, each for several test cases.
Writes results/torch_{dynamic_raw,dynamic_counts,summary}.csv.

usage: venv/bin/python profile_torch.py [--jobs N] [--testcases K] [--steps S] [--batch B]
"""
import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from profile_opcodes import BUILD, QEMU, RESULTS, ROOT, annotate

ROOTFS = ROOT / "third_party/rv-rootfs"
PLUGIN = ROOT / "tools/opcount_roi.so"
SCRIPT = ROOT / "torch_cnn/mnist_cnn.py"
GUEST_LIBDIR = "/usr/lib/riscv64-linux-gnu"

MODES = ["infer", "train"]
BLAS = {
    "openblas": f"{GUEST_LIBDIR}/openblas-pthread",
    "reference": f"{GUEST_LIBDIR}/blas:{GUEST_LIBDIR}/lapack",
}
CPUS = {"rvv": "max", "novec": "max,v=false"}
CONFIG_COLS = ["mode", "blas", "cpu", "config"]

# Shared objects grouped for reporting; anything else is "other".
OBJECT_GROUPS = [
    ("libtorch_cpu", "torch (libtorch_cpu)"),
    ("libc10", "torch (libc10)"),
    ("libtorch_python", "torch (python bindings)"),
    ("libopenblas", "BLAS"),
    ("libblas", "BLAS"),
    ("liblapack", "BLAS"),
    ("python3", "CPython interpreter"),
    ("libpython", "CPython interpreter"),
    ("_multiarray", "numpy"),
    ("libc.so", "libc"),
    ("libm.so", "libm"),
    ("libgomp", "OpenMP runtime"),
]


def object_group(obj):
    name = obj.rsplit("/", 1)[-1]
    for needle, group in OBJECT_GROUPS:
        if needle in name:
            return group
    return "other"


def run_one(cfg, k, args):
    mode, blas, cpu = cfg
    config = f"{mode}_{blas}_{cpu}"
    outdir = BUILD / "torch" / config
    outdir.mkdir(parents=True, exist_ok=True)
    out, maps, log = (outdir / f"tc{k}{ext}" for ext in (".csv", ".maps", ".log"))
    if not out.exists():
        tmp = out.with_suffix(".tmp")
        env = dict(os.environ, ROI_MAPS=str(maps), OPENBLAS_NUM_THREADS="1",
                   OMP_NUM_THREADS="1", LD_LIBRARY_PATH=BLAS[blas])
        cmd = [str(QEMU), "-L", str(ROOTFS), "-cpu", CPUS[cpu],
               "-plugin", f"{PLUGIN},outfile={tmp},maps={maps}",
               str(ROOTFS / "usr/bin/python3"), str(SCRIPT),
               "--mode", mode, "--steps", str(args.steps), "--batch", str(args.batch),
               "--seed", str(k + 1), "--offset", str(k * 1000)]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        log.write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            raise RuntimeError(f"{config} tc{k} failed, see {log}:\n{r.stderr[-2000:]}")
        tmp.rename(out)
    print(f"  done {config} tc{k}", flush=True)
    df = pd.read_csv(out, keep_default_na=False)
    df["object"] = df["object"].str.rsplit("/", n=1).str[-1]
    return df.assign(mode=mode, blas=blas, cpu=cpu, config=config, testcase=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--testcases", type=int, default=5)
    ap.add_argument("--steps", type=int, default=5, help="measured iterations per run")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    for p in (QEMU, PLUGIN, ROOTFS / "usr/bin/python3"):
        if not p.exists():
            sys.exit(f"missing {p}: run tools/build_qemu.sh, make -C tools, tools/build_rootfs.sh")

    jobs = [(cfg, k) for cfg in itertools.product(MODES, BLAS, CPUS)
            for k in range(args.testcases)]
    print(f"running {len(jobs)} QEMU runs on {args.jobs} workers "
          f"(~2 min each, mostly python startup + import torch)", flush=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        raw = pd.concat(pool.map(lambda j: run_one(*j, args), jobs), ignore_index=True)
    raw["object_group"] = raw["object"].map(object_group)
    raw = annotate(raw)
    RESULTS.mkdir(exist_ok=True)
    raw.to_csv(RESULTS / "torch_dynamic_raw.csv", index=False)

    # mean/std over test cases per (config, scope, object_group, mnemonic); "ALL" = every object.
    keys = CONFIG_COLS + ["scope", "mnemonic"]
    per_tc = pd.concat([
        raw.groupby(keys + ["object_group", "testcase"], as_index=False)["count"].sum(),
        raw.groupby(keys + ["testcase"], as_index=False)["count"].sum().assign(object_group="ALL"),
    ])
    wide = per_tc.pivot_table(index=keys + ["object_group"], columns="testcase",
                              values="count", fill_value=0)
    dyn = pd.DataFrame({"mean": wide.mean(axis=1), "std": wide.std(axis=1, ddof=0)}).reset_index()
    dyn["fraction"] = dyn["mean"] / dyn.groupby(
        CONFIG_COLS + ["scope", "object_group"])["mean"].transform("sum")
    dyn = annotate(dyn)
    dyn.to_csv(RESULTS / "torch_dynamic_counts.csv", index=False)

    # One row per config: totals and ROI share per object group.
    tot = raw.groupby(CONFIG_COLS + ["scope", "testcase"])["count"].sum().unstack("scope")
    agg = tot.groupby(level=CONFIG_COLS).agg(["mean", "std"])
    summary = pd.DataFrame({
        "all_total_mean": agg[("all", "mean")],
        "roi_total_mean": agg[("roi", "mean")],
        "roi_total_std": agg[("roi", "std")],
    })
    summary["roi_per_image"] = summary["roi_total_mean"] / (args.steps * args.batch)
    shares = (raw[raw.scope == "roi"].groupby(CONFIG_COLS + ["object_group"])["count"].sum()
              .unstack("object_group").fillna(0))
    shares = shares.div(shares.sum(axis=1), axis=0).add_prefix("roi_share_")
    summary = summary.join(shares).reset_index()
    summary.to_csv(RESULTS / "torch_summary.csv", index=False)
    print(summary[["config", "all_total_mean", "roi_total_mean", "roi_per_image"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
