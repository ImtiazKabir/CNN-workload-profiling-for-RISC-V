"""Slide-sized opcode plots (large type, few variants) -> results/plots/slides/*.png

Single-panel figures carry no title: the slide heading names them.

Reads the CSVs written by profile_opcodes.py and profile_torch.py.
usage: venv/bin/python plot_slides.py
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot import CATEGORY_GROUPS, GROUP_ORDER, PLOTS, RESULTS, SEQ, SERIES, SURFACE, INK2

OUT = PLOTS / "slides"
FIGSIZE = (16, 6.9)  # fills a 1664x720 slide area

plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.labelsize": 17,
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16})

# C variants shown on slides: label -> variant id
C_ROWS = {
    "-O0": "O0_rv64gc", "-O1": "O1_rv64gc", "-O2": "O2_rv64gc", "-O3": "O3_rv64gc",
    "-Os": "O3_rv64gc".replace("O3", "Os"), "-O3 +Zba/Zbb/Zbs": "O3_rv64gc_zba_zbb_zbs",
    "-O3 +V": "O3_rv64gcv", "-O3 +V +fast-math": "O3_rv64gcv_fastmath",
}
TORCH_ROWS = {
    "infer · OpenBLAS + RVV": "infer_openblas_rvv", "infer · OpenBLAS scalar": "infer_openblas_novec",
    "infer · reference BLAS": "infer_reference_novec", "train · OpenBLAS + RVV": "train_openblas_rvv",
    "train · OpenBLAS scalar": "train_openblas_novec", "train · reference BLAS": "train_reference_novec",
}
MARCH_LABELS = {"rv64g": "rv64g", "rv64gc": "rv64gc", "rv64gc_zba_zbb_zbs": "rv64gc + Zba/Zbb/Zbs",
                "rv64gcv": "rv64gcv"}


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / name)


def shares(df, value, by="canonical"):
    s = df.groupby(by)[value].sum()
    return s / s.sum()


def pct_axis(ax, axis="x"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def heatmap(rows, name, title, top=14):
    """rows: label -> share Series. Columns = the top opcodes by max share over rows."""
    m = pd.DataFrame(rows).T.fillna(0)
    cols = m.max().sort_values(ascending=False).index[:top]
    m = m[cols]
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.imshow(m.to_numpy(), cmap=SEQ, aspect="auto", vmin=0, vmax=max(0.3, m.to_numpy().max()))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=15,
                        color="#fcfcfb" if v > 0.17 else "#0b0b0b")
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right", fontsize=17)
    ax.set_yticks(range(len(m)), m.index, fontsize=17)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, name)


def grouped_hbar(series, name, title, top=15, colors=SERIES):
    """series: label -> share Series; bars for the top opcodes by the first series."""
    first = next(iter(series.values()))
    names = first.sort_values(ascending=False).index[:top]
    n = len(series)
    h = 0.8 / n
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    y = np.arange(len(names))
    for i, (label, s) in enumerate(series.items()):
        ax.barh(y + (i - (n - 1) / 2) * h, s.reindex(names).fillna(0), h, color=colors[i], label=label)
    ax.set_yticks(y, names, fontsize=17)
    ax.invert_yaxis()
    pct_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, name)


def category_stack(rows, name, title):
    """rows: label -> DataFrame with category + value column 'v'."""
    mix = pd.DataFrame({k: df.assign(g=df.category.map(CATEGORY_GROUPS).fillna(GROUP_ORDER[-1]))
                        .groupby("g")["v"].sum() for k, df in rows.items()}).T.fillna(0)
    mix = mix.div(mix.sum(axis=1), axis=0).reindex(columns=GROUP_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    y = np.arange(len(mix))
    left = np.zeros(len(mix))
    for i, g in enumerate(GROUP_ORDER):
        ax.barh(y, mix[g], 0.72, left=left, color=SERIES[i], label=g, edgecolor=SURFACE, linewidth=1.5)
        for yi, (l, w) in enumerate(zip(left, mix[g])):
            if w >= 0.06:
                ax.text(l + w / 2, yi, f"{w * 100:.0f}", ha="center", va="center", fontsize=14,
                        color="#fcfcfb" if g in ("int ALU", "FP convert/move", "vector") else "#0b0b0b")
        left += mix[g].to_numpy()
    ax.set_yticks(y, mix.index, fontsize=17)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    pct_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=4, fontsize=15)
    save(fig, name)


def c_slides(dyn, static, summary):
    for prog in ["axel", "magicalbat"]:
        d = dyn[(dyn.program == prog) & (dyn.region == "all")]
        heatmap({k: shares(d[d.variant == v], "mean") for k, v in C_ROWS.items()},
                f"{prog}_dyn_heatmap.png", f"{prog}: executed opcode mix per compiler variant")
        category_stack({k: d[d.variant == v].rename(columns={"mean": "v"}) for k, v in C_ROWS.items()},
                       f"{prog}_dyn_category.png", f"{prog}: executed instruction classes")
        s = static[(static.program == prog) & (static.source == "obj")]
        grouped_hbar({"executed (dynamic)": shares(d[d.variant == "O2_rv64gc"], "mean"),
                      "in the binary (static)": shares(s[s.variant == "O2_rv64gc"], "count")},
                     f"{prog}_static_vs_dynamic.png",
                     f"{prog}, -O2 rv64gc: top-15 executed opcodes vs their share of the code")

    # Total executed instructions per opt level and march (default FP semantics).
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, layout="constrained")
    opts = ["O0", "O1", "O2", "O3", "Os"]
    x = np.arange(len(opts))
    w = 0.8 / len(MARCH_LABELS)
    for ax, prog in zip(axes, ["axel", "magicalbat"]):
        s = summary[(summary.program == prog) & ~summary.fastmath]
        for i, march in enumerate(MARCH_LABELS):
            v = s[s.march == march].set_index("opt").reindex(opts).dyn_total_mean / 1e9
            ax.bar(x + (i - 1.5) * w, v, w, color=SERIES[i], label=MARCH_LABELS[march],
                   edgecolor=SURFACE, linewidth=1)
        ax.set_xticks(x, ["-" + o for o in opts])
        ax.grid(axis="x", visible=False)
        ax.set_title(prog, loc="left", color=INK2)
        ax.set_ylabel("executed instructions (billions)")
    axes[1].legend(loc="upper right")
    save(fig, "c_totals.png")

    # Vector spotlight: counts at -O3 with and without V, program code only.
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, layout="constrained")
    for ax, prog in zip(axes, ["axel", "magicalbat"]):
        d = dyn[(dyn.program == prog) & (dyn.region == "app")]
        a = d[d.variant == "O3_rv64gc"].groupby("canonical")["mean"].sum() / 1e6
        b = d[d.variant == "O3_rv64gcv"].groupby("canonical")["mean"].sum() / 1e6
        names = pd.concat([a, b], axis=1).fillna(0).max(axis=1).sort_values(ascending=False).index[:12]
        y = np.arange(len(names))
        ax.barh(y - 0.2, a.reindex(names).fillna(0), 0.4, color=SERIES[0], label="-O3 rv64gc")
        ax.barh(y + 0.2, b.reindex(names).fillna(0), 0.4, color=SERIES[1], label="-O3 rv64gcv")
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("executed (millions), program code")
        ax.set_title(f"{prog}: {a.sum():,.0f}M → {b.sum():,.0f}M", loc="left", color=INK2)
    axes[1].legend(loc="lower right")
    save(fig, "c_vector.png")


def torch_slides(t, summary):
    roi = t[(t.scope == "roi")]
    allobj = roi[roi.object_group == "ALL"]
    grouped_hbar({"OpenBLAS + RVV": shares(allobj[allobj.config == "train_openblas_rvv"], "mean"),
                  "reference BLAS": shares(allobj[allobj.config == "train_reference_novec"], "mean")},
                 "torch_top_opcodes.png", "PyTorch CNN, training steps: top-15 executed opcodes", top=15)
    c = roi[roi.config == "train_openblas_rvv"]
    heatmap({"libtorch_cpu": shares(c[c.object_group == "torch (libtorch_cpu)"], "mean"),
             "OpenBLAS (RVV)": shares(c[c.object_group == "BLAS"], "mean"),
             "libc": shares(c[c.object_group == "libc"], "mean"),
             "all code": shares(c[c.object_group == "ALL"], "mean")},
            "torch_library_heatmap.png", "PyTorch training, OpenBLAS + RVV: opcode mix inside each library")
    category_stack({k: allobj[allobj.config == v].rename(columns={"mean": "v"}) for k, v in TORCH_ROWS.items()},
                   "torch_category.png", "PyTorch CNN: executed instruction classes")

    # Instructions per image, stacked by library.
    images = (summary.roi_total_mean / summary.roi_per_image).iloc[0]
    fold = {"torch (libtorch_cpu)": "libtorch_cpu", "BLAS": "BLAS", "libc": "libc",
            "CPython interpreter": "CPython"}
    d = roi[roi.object_group != "ALL"].assign(lib=lambda x: x.object_group.map(fold).fillna("other"))
    m = d.groupby(["config", "lib"])["mean"].sum().unstack("lib", fill_value=0) / images / 1e6
    order = ["libtorch_cpu", "BLAS", "libc", "CPython", "other"]
    m = m.reindex(list(TORCH_ROWS.values())).reindex(columns=order, fill_value=0)
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    y = np.arange(len(m))
    left = np.zeros(len(m))
    for i, lib in enumerate(order):
        ax.barh(y, m[lib], 0.7, left=left, color=SERIES[i], label=lib, edgecolor=SURFACE, linewidth=1.5)
        left += m[lib].to_numpy()
    for yi, total in zip(y, left):
        ax.text(total, yi, f"  {total:.2f}M", va="center", fontsize=16)
    ax.set_yticks(y, list(TORCH_ROWS))
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("executed instructions per image (millions)")
    ax.legend(loc="lower right")
    save(fig, "torch_per_image.png")


# ISA-pruning candidates (CARVE-V style): instruction family -> regex on the canonical mnemonic,
# and an illustrative extra cost in cycles when it is emulated by a trap handler instead of
# executed in hardware (assumptions, in the spirit of the "hardware ~20 / emulated ~200" example).
PRUNE_GROUPS = {
    "atomics (A)": (r"^(?:amo|lr\.|sc\.)", 40),
    "mulh / mulhu / mulhsu": (r"^mulh", 60),
    "integer div / rem": (r"^(?:div|rem)", 180),
    "FP div / sqrt": (r"^f(?:div|sqrt)\.", 300),
    "FP min / max / class": (r"^f(?:min|max|class)\.", 40),
    "double precision (D)": (r"(?:\.d$|\.d\.|^fld$|^fsd$)", 100),
    "fmsub / fnmadd / fnmsub": (r"^f(?:nmadd|nmsub|msub)\.", 30),
    "fence": (r"^fence", 30),
}
PRUNE_WORKLOADS = ["axel (C CNN)", "magicalbat (C MLP)", "PyTorch inference", "PyTorch training"]


def prune_frequencies(dyn, t):
    """ppm of executed instructions per candidate family, rv64gc baseline (CVA6 has no V)."""
    sources = {
        "axel (C CNN)": dyn[(dyn.program == "axel") & (dyn.variant == "O2_rv64gc") & (dyn.region == "all")],
        "magicalbat (C MLP)": dyn[(dyn.program == "magicalbat") & (dyn.variant == "O2_rv64gc")
                                  & (dyn.region == "all")],
        "PyTorch inference": t[(t.config == "infer_openblas_novec") & (t.scope == "roi") & (t.object_group == "ALL")],
        "PyTorch training": t[(t.config == "train_openblas_novec") & (t.scope == "roi") & (t.object_group == "ALL")],
    }
    ppm = {}
    for w, df in sources.items():
        total = df["mean"].sum()
        ppm[w] = {g: df[df.canonical.str.contains(rx)]["mean"].sum() / total * 1e6
                  for g, (rx, _) in PRUNE_GROUPS.items()}
    return pd.DataFrame(ppm)[PRUNE_WORKLOADS]


def prune_slides(dyn, t):
    ppm = prune_frequencies(dyn, t)
    print(ppm.round(1).to_string())

    # Heatmap of measured frequency, log colour scale.
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    v = np.log10(ppm.to_numpy() + 0.1)
    ax.imshow(v, cmap=SEQ, aspect="auto", vmin=-1, vmax=5.5)
    for i in range(ppm.shape[0]):
        for j in range(ppm.shape[1]):
            p = ppm.iat[i, j]
            label = "never" if p == 0 else f"{p / 1e4:.1f}%" if p >= 1e4 else f"{p:,.0f} ppm" if p >= 1 else "<1 ppm"
            ax.text(j, i, label, ha="center", va="center", fontsize=17,
                    color="#fcfcfb" if v[i, j] > 3.2 else "#0b0b0b")
    ax.set_xticks(range(ppm.shape[1]), ppm.columns, fontsize=17)
    ax.xaxis.tick_top()
    ax.set_yticks(range(ppm.shape[0]), ppm.index, fontsize=17)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "prune_frequency.png")

    # Estimated bridge overhead = frequency x extra cycles, for the recommended prune set.
    prune_set = ["atomics (A)", "mulh / mulhu / mulhsu", "integer div / rem", "FP div / sqrt",
                 "FP min / max / class"]
    extra = pd.Series({g: c for g, (_, c) in PRUNE_GROUPS.items()})
    cost = ppm.mul(extra, axis=0) / 1e6 * 100  # % extra cycles at ~1 cycle/instruction
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    x = np.arange(len(PRUNE_WORKLOADS))
    bottom = np.zeros(len(x))
    for i, g in enumerate(prune_set):
        ax.bar(x, cost.loc[g], 0.55, bottom=bottom, color=SERIES[i], label=g, edgecolor=SURFACE, linewidth=1.5)
        bottom += cost.loc[g].to_numpy()
    for xi, b in zip(x, bottom):
        ax.text(xi, b + 0.15, f"+{b:.1f}%", ha="center", va="bottom", fontsize=19)
    print("D alone (% extra cycles):", cost.loc["double precision (D)"].round(1).to_dict())
    ax.set_ylim(0, bottom.max() * 1.15)
    ax.set_xticks(x, PRUNE_WORKLOADS)
    ax.set_ylabel("estimated extra cycles (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=16)
    save(fig, "prune_cost.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prune_slides(pd.read_csv(RESULTS / "dynamic_counts.csv"), pd.read_csv(RESULTS / "torch_dynamic_counts.csv"))
    c_slides(pd.read_csv(RESULTS / "dynamic_counts.csv"), pd.read_csv(RESULTS / "static_counts.csv"),
             pd.read_csv(RESULTS / "summary.csv"))
    torch_slides(pd.read_csv(RESULTS / "torch_dynamic_counts.csv"), pd.read_csv(RESULTS / "torch_summary.csv"))


if __name__ == "__main__":
    main()
