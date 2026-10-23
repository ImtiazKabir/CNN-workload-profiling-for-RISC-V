"""Plots from results/torch_*.csv (written by profile_torch.py) -> results/plots/torch_*.png

usage: venv/bin/python plot_torch.py
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot import CATEGORY_GROUPS, GROUP_ORDER, PLOTS, RESULTS, SERIES, SURFACE, save

# Object groups (profile_torch.py) folded to <= 8 stack slots.
LIB_FOLD = {
    "torch (libtorch_cpu)": "libtorch_cpu (kernels + dispatch)",
    "BLAS": "BLAS",
    "libc": "libc (memcpy/memset/malloc)",
    "CPython interpreter": "CPython interpreter",
    "torch (libc10)": "torch runtime (c10 + bindings)",
    "torch (python bindings)": "torch runtime (c10 + bindings)",
}
LIB_ORDER = ["libtorch_cpu (kernels + dispatch)", "BLAS", "libc (memcpy/memset/malloc)",
             "CPython interpreter", "torch runtime (c10 + bindings)", "other"]


def config_order(df):
    keys = df[["config", "mode", "blas", "cpu"]].drop_duplicates()
    return list(keys.sort_values(["mode", "blas", "cpu"]).config)


def plot_by_library(dyn, images):
    """ROI instructions per image, stacked by shared object."""
    d = dyn[(dyn.scope == "roi") & (dyn.object_group != "ALL")]
    d = d.assign(lib=d.object_group.map(LIB_FOLD).fillna("other"))
    m = d.groupby(["config", "lib"])["mean"].sum().unstack("lib", fill_value=0)
    m = (m / images / 1e6).reindex(config_order(d)).reindex(columns=LIB_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(m) + 1.5))
    y = np.arange(len(m))
    left = np.zeros(len(m))
    for i, lib in enumerate(LIB_ORDER):
        ax.barh(y, m[lib], 0.7, left=left, color=SERIES[i], label=lib,
                edgecolor=SURFACE, linewidth=1)
        left += m[lib].to_numpy()
    for yi, total in zip(y, left):
        ax.text(total, yi, f" {total:.2f}M", va="center", fontsize=8)
    ax.set_yticks(y, m.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("executed instructions per image, millions (region of interest)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("PyTorch MNIST CNN: where the instructions go", loc="left")
    fig.tight_layout()
    save(fig, "torch_roi_by_library.png")


def plot_category_mix(dyn):
    """Instruction-class mix of the ROI per config, and per library for each config."""
    d = dyn[dyn.scope == "roi"]
    d = d.assign(group=d.category.map(CATEGORY_GROUPS).fillna(GROUP_ORDER[-1]))
    rows = []
    for config in config_order(d):
        c = d[d.config == config]
        rows.append((config, c[c.object_group == "ALL"]))
        for lib in ["torch (libtorch_cpu)", "BLAS", "libc"]:
            rows.append((f"    {lib}", c[c.object_group == lib]))
    mix = pd.DataFrame([r.groupby("group")["mean"].sum() for _, r in rows],
                       index=[name for name, _ in rows]).fillna(0)
    mix = mix.div(mix.sum(axis=1), axis=0).reindex(columns=GROUP_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 0.24 * len(mix) + 1.5))
    y = np.arange(len(mix))
    left = np.zeros(len(mix))
    for i, g in enumerate(GROUP_ORDER):
        ax.barh(y, mix[g], 0.8, left=left, color=SERIES[i], label=g,
                edgecolor=SURFACE, linewidth=1)
        left += mix[g].to_numpy()
    ax.set_yticks(y, mix.index)
    for label in ax.get_yticklabels():
        if not label.get_text().startswith(" "):
            label.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("PyTorch MNIST CNN: instruction class mix (ROI; bold = all code)", loc="left")
    fig.tight_layout()
    save(fig, "torch_category_mix.png")


def plot_top(dyn, config, top=25):
    d = dyn[(dyn.scope == "roi") & (dyn.config == config) & (dyn.object_group == "ALL")]
    d = d.nlargest(top, "mean")
    fig, ax = plt.subplots(figsize=(7, 0.26 * top + 1.2))
    y = np.arange(len(d))
    ax.barh(y, d.fraction, 0.7, color=SERIES[0])
    ax.set_yticks(y, d.mnemonic)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{config}: top-{top} executed opcodes (ROI)", loc="left")
    fig.tight_layout()
    save(fig, f"torch_top{top}_{config}.png")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    dyn = pd.read_csv(RESULTS / "torch_dynamic_counts.csv")
    summary = pd.read_csv(RESULTS / "torch_summary.csv")
    images = (summary.roi_total_mean / summary.roi_per_image).iloc[0]
    plot_by_library(dyn, images)
    plot_category_mix(dyn)
    for config in ["train_openblas_rvv", "train_reference_novec", "infer_openblas_rvv"]:
        plot_top(dyn, config)


if __name__ == "__main__":
    main()
