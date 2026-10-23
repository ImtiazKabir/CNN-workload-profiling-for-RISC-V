"""Plots from results/*.csv (written by profile_opcodes.py) -> results/plots/*.png

usage: venv/bin/python plot.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"

OPTS = ["O0", "O1", "O2", "O3", "Os"]
MARCHES = ["rv64g", "rv64gc", "rv64gc_zba_zbb_zbs", "rv64gcv"]

# Reference categorical palette (fixed order), text inks, surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])

# Fine categories (profile_opcodes.py) folded to <= 8 stack slots.
CATEGORY_GROUPS = {
    "int-alu": "int ALU", "load": "int load/store", "store": "int load/store",
    "branch": "branch/jump", "jump": "branch/jump", "fp-arith": "FP arith + FMA",
    "fp-fma": "FP arith + FMA", "fp-ld/st": "FP load/store",
    "fp-cvt/move": "FP convert/move", "vector": "vector",
}
GROUP_ORDER = ["int ALU", "int load/store", "branch/jump", "FP arith + FMA",
               "FP load/store", "FP convert/move", "vector",
               "other (mul/div, bitmanip, system)"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "legend.frameon": False,
})


def variant_order(df):
    """Rows ordered march-major, then opt, then fast-math."""
    keys = df[["variant", "march", "opt", "fastmath"]].drop_duplicates()
    keys = keys.assign(m=keys.march.map(MARCHES.index), o=keys.opt.map(OPTS.index))
    return list(keys.sort_values(["m", "o", "fastmath"]).variant)


def save(fig, name):
    fig.savefig(PLOTS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", PLOTS / name)


def plot_totals(summary, prog):
    """Dynamic instruction count per variant, stacked app (program) vs lib (libc/libm/ld.so)."""
    s = summary[summary.program == prog]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    x = np.arange(len(OPTS))
    for row, fm in enumerate([False, True]):
        for col, march in enumerate(MARCHES):
            ax = axes[row, col]
            d = s[(s.march == march) & (s.fastmath == fm)].set_index("opt").reindex(OPTS)
            app, lib = d.dyn_app_mean / 1e9, d.dyn_lib_mean / 1e9
            ax.bar(x, app, 0.6, color=SERIES[0], label="program code", edgecolor=SURFACE, linewidth=1)
            ax.bar(x, lib, 0.6, bottom=app, color=SERIES[1], label="libc / libm / ld.so",
                   edgecolor=SURFACE, linewidth=1)
            ax.errorbar(x, app + lib, yerr=d.dyn_total_std / 1e9, fmt="none", ecolor=INK2, capsize=2)
            ax.set_xticks(x, OPTS)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{march}{' + fast-math' if fm else ''}", loc="left")
            if col == 0:
                ax.set_ylabel("executed instructions (billions)")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(f"{prog}: dynamic instruction count (mean of test cases, ±std)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, f"{prog}_dynamic_totals.png")


def plot_static_size(summary, prog):
    """Static instruction count of program code (objdump of the .o files) per variant."""
    s = summary[summary.program == prog]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    x = np.arange(len(OPTS))
    w = 0.8 / len(MARCHES)
    for ax, fm in zip(axes, [False, True]):
        for i, march in enumerate(MARCHES):
            d = s[(s.march == march) & (s.fastmath == fm)].set_index("opt").reindex(OPTS)
            ax.bar(x + (i - 1.5) * w, d.static_obj_total, w, color=SERIES[i], label=march,
                   edgecolor=SURFACE, linewidth=1)
        ax.set_xticks(x, OPTS)
        ax.grid(axis="x", visible=False)
        ax.set_title("fast-math" if fm else "default FP semantics", loc="left")
    axes[0].set_ylabel("static instructions in program .o files")
    axes[1].legend(loc="upper left", ncols=2)
    fig.suptitle(f"{prog}: static code size", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, f"{prog}_static_size.png")


def plot_category_mix(dyn, static, prog):
    """Instruction-class mix per variant, dynamic (all code) next to static (program .o)."""
    def mix(df, value):
        df = df.assign(group=df.category.map(CATEGORY_GROUPS).fillna(GROUP_ORDER[-1]))
        m = df.groupby(["variant", "group"])[value].sum().unstack("group", fill_value=0)
        m = m.div(m.sum(axis=1), axis=0)
        return m.reindex(columns=GROUP_ORDER, fill_value=0)

    d = dyn[(dyn.program == prog) & (dyn.region == "all")]
    s = static[(static.program == prog) & (static.source == "obj")]
    order = variant_order(d)
    mixes = [("dynamic (executed, incl. libc)", mix(d, "mean").reindex(order)),
             ("static (program .o)", mix(s, "count").reindex(order))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.22 * len(order) + 1.5), sharey=True)
    y = np.arange(len(order))
    for ax, (title, m) in zip(axes, mixes):
        left = np.zeros(len(order))
        for i, g in enumerate(GROUP_ORDER):
            ax.barh(y, m[g], 0.8, left=left, color=SERIES[i], label=g, edgecolor=SURFACE, linewidth=1)
            left += m[g].to_numpy()
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left")
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(f"{prog}: instruction class mix", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, f"{prog}_category_mix.png")


def plot_heatmap(df, value, prog, kind, top=30):
    """Variants x top-N canonical mnemonics, share of all instructions."""
    m = df.groupby(["variant", "canonical"])[value].sum().unstack("canonical", fill_value=0)
    m = m.div(m.sum(axis=1), axis=0)
    cols = m.mean().sort_values(ascending=False).index[:top]
    m = m.reindex(variant_order(df))[cols]
    fig, ax = plt.subplots(figsize=(0.38 * top + 3, 0.22 * len(m) + 2))
    im = ax.imshow(m.to_numpy() * 100, cmap=SEQ, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right")
    ax.set_yticks(range(len(m)), m.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("% of instructions")
    cb.outline.set_visible(False)
    what = "dynamic (executed, incl. libc)" if kind == "dynamic" else "static (program .o)"
    ax.set_title(f"{prog}: top-{top} opcodes, {what}; aliases resolved, RVC folded", loc="left")
    fig.tight_layout()
    save(fig, f"{prog}_heatmap_{kind}.png")


def plot_top(dyn, static, prog, variant, top=20):
    """Top-N opcodes of one variant: static share vs dynamic share."""
    d = dyn[(dyn.program == prog) & (dyn.region == "all") & (dyn.variant == variant)]
    s = static[(static.program == prog) & (static.source == "obj") & (static.variant == variant)]
    ds = d.groupby("mnemonic")["mean"].sum()
    ss = s.groupby("mnemonic")["count"].sum()
    ds, ss = ds / ds.sum(), ss / ss.sum()
    names = ds.sort_values(ascending=False).index[:top]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 0.28 * top + 1.2))
    ax.barh(y - 0.2, ds.reindex(names), 0.4, color=SERIES[0], label="dynamic (executed)")
    ax.barh(y + 0.2, ss.reindex(names).fillna(0), 0.4, color=SERIES[1], label="static (program .o)")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title(f"{prog} {variant}: top-{top} executed opcodes", loc="left")
    fig.tight_layout()
    save(fig, f"{prog}_top{top}_{variant}.png")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RESULTS / "summary.csv")
    dyn = pd.read_csv(RESULTS / "dynamic_counts.csv")
    static = pd.read_csv(RESULTS / "static_counts.csv")
    for prog in sorted(summary.program.unique()):
        plot_totals(summary, prog)
        plot_static_size(summary, prog)
        plot_category_mix(dyn, static, prog)
        base = ~dyn.fastmath
        plot_heatmap(dyn[(dyn.program == prog) & (dyn.region == "all") & base], "mean", prog, "dynamic")
        plot_heatmap(static[(static.program == prog) & (static.source == "obj") & ~static.fastmath],
                     "count", prog, "static")
        for variant in ["O0_rv64gc", "O2_rv64gc", "O3_rv64gcv"]:
            plot_top(dyn, static, prog, variant)


if __name__ == "__main__":
    main()
