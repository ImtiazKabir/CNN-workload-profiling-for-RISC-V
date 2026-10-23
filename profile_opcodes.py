"""Static + dynamic RISC-V opcode profiling of the MNIST programs in CNN/.

For every (program, opt, march, fast-math) variant:
  * compile each .c to .s (counted: "asm"), assemble to .o (objdump: "obj"), link;
  * run the binary under QEMU with tools/opcount.so for several test cases.
Writes results/{static_counts,dynamic_raw,dynamic_counts,summary}.csv.

usage: venv/bin/python profile_opcodes.py [--jobs N] [--testcases K] [--quick] [--program axel ...]
"""
import argparse
import itertools
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
RESULTS = ROOT / "results"
QEMU = ROOT / "third_party/qemu/build/qemu-riscv64"
PLUGIN = ROOT / "tools/opcount.so"
SYSROOT = "/usr/riscv64-linux-gnu"
CC = "riscv64-linux-gnu-gcc"
OBJDUMP = "riscv64-linux-gnu-objdump"

OPTS = ["O0", "O1", "O2", "O3", "Os"]
MARCHES = ["rv64g", "rv64gc", "rv64gc_zba_zbb_zbs", "rv64gcv"]
FASTMATH = [False, True]

# Each test case uses a different RNG seed and a different slice of MNIST.
PROGRAMS = {
    "axel": dict(
        src=ROOT / "CNN/AxelMontlahuc",
        patch=ROOT / "patches/axel.patch",
        sources=["main.c", "lib/import.c", "lib/convolution.c", "lib/pooling.c",
                 "lib/dense.c", "lib/output.c", "lib/backprop.c"],
        rundir=ROOT / "CNN/AxelMontlahuc",  # reads ./MNIST/*
        # n_train n_test epochs seed offset
        args=lambda k: [300, 100, 1, k + 1, k * 1000],
    ),
    "magicalbat": dict(
        src=ROOT / "CNN/magicalbat",
        patch=ROOT / "patches/magicalbat.patch",
        sources=["main.c"],  # arena.c / prng.c are #included
        rundir=BUILD / "data/magicalbat",  # *.mat from tools/make_mat.py
        # n_train n_test epochs batch_size seed offset
        args=lambda k: [2000, 500, 1, 50, k + 1, k * 1000],
    ),
}

VARIANT_COLS = ["program", "opt", "march", "fastmath", "variant"]

# 1:1 assembler aliases -> real instruction, so names from GCC (.s), objdump
# and QEMU's disassembler line up.
ALIASES = {
    "li": "addi", "mv": "addi", "nop": "addi", "not": "xori", "neg": "sub",
    "negw": "subw", "sext.w": "addiw", "seqz": "sltiu", "snez": "sltu",
    "sltz": "slt", "sgtz": "slt", "beqz": "beq", "bnez": "bne", "blez": "bge",
    "bgez": "bge", "bltz": "blt", "bgtz": "blt", "bgt": "blt", "ble": "bge",
    "bgtu": "bltu", "bleu": "bgeu", "j": "jal", "jr": "jalr", "ret": "jalr",
    "call": "jalr", "tail": "jalr", "la": "auipc", "lla": "auipc",
    "zext.b": "andi", "zext.w": "add.uw",
    "fmv.s": "fsgnj.s", "fmv.d": "fsgnj.d", "fneg.s": "fsgnjn.s",
    "fneg.d": "fsgnjn.d", "fabs.s": "fsgnjx.s", "fabs.d": "fsgnjx.d",
    "frflags": "csrrs", "fsflags": "csrrw", "frcsr": "csrrs", "fscsr": "csrrw",
    "frrm": "csrrs", "fsrm": "csrrw", "csrr": "csrrs", "csrw": "csrrw",
    "csrs": "csrrs", "csrc": "csrrc", "rdcycle": "csrrs", "rdtime": "csrrs",
}

BITMANIP = {
    "sh1add", "sh2add", "sh3add", "add.uw", "sh1add.uw", "sh2add.uw",
    "sh3add.uw", "slli.uw", "andn", "orn", "xnor", "clz", "clzw", "ctz",
    "ctzw", "cpop", "cpopw", "max", "maxu", "min", "minu", "sext.b", "sext.h",
    "zext.h", "rol", "rolw", "ror", "rori", "roriw", "rorw", "orc.b", "rev8",
    "bclr", "bclri", "bext", "bexti", "binv", "binvi", "bset", "bseti",
}

CATEGORY_ORDER = [
    "int-alu", "mul/div", "load", "store", "branch", "jump", "fp-arith",
    "fp-fma", "fp-ld/st", "fp-cvt/move", "bitmanip", "vector", "atomic",
    "system", "other",
]


def canonical(mnemonic):
    base = mnemonic[2:] if mnemonic.startswith("c.") else mnemonic
    return ALIASES.get(base, base)


def category(canon):
    m = canon
    if m.startswith("v"):
        return "vector"
    if m in BITMANIP:
        return "bitmanip"
    if m in {"lb", "lh", "lw", "ld", "lbu", "lhu", "lwu"}:
        return "load"
    if m in {"sb", "sh", "sw", "sd"}:
        return "store"
    if m in {"beq", "bne", "blt", "bge", "bltu", "bgeu"}:
        return "branch"
    if m in {"jal", "jalr"}:
        return "jump"
    if re.match(r"(mul|div|rem)", m):
        return "mul/div"
    if m in {"flw", "fld", "fsw", "fsd", "flh", "fsh"}:
        return "fp-ld/st"
    if re.match(r"f(n?madd|n?msub)\.", m):
        return "fp-fma"
    if re.match(r"f(cvt|mv|sgnj|class)", m):
        return "fp-cvt/move"
    if re.match(r"f(add|sub|mul|div|sqrt|min|max|eq|lt|le)\.", m):
        return "fp-arith"
    if re.match(r"(lr|sc|amo)", m):
        return "atomic"
    if m in {"ecall", "ebreak", "fence", "fence.i"} or m.startswith("csr"):
        return "system"
    if re.match(r"(addi?w?|subw?|lui|auipc|andi?|ori?|xori?|s[lr][la]i?w?|"
                r"slti?u?)$", m):
        return "int-alu"
    return "other"


def annotate(df):
    df["base_mnemonic"] = df["mnemonic"].str.removeprefix("c.")
    df["canonical"] = df["mnemonic"].map(canonical)
    df["category"] = df["canonical"].map(category)
    return df


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed:\n{r.stderr[-2000:]}")
    return r


def prepare_sources(name, prog):
    dst = BUILD / "src" / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(prog["src"], dst, ignore=shutil.ignore_patterns(".git", "MNIST"))
    run(["patch", "-p2", "-s", "-i", str(prog["patch"])], cwd=dst)
    return dst


def variant_id(opt, march, fm):
    return f"{opt}_{march}" + ("_fastmath" if fm else "")


def build_variant(name, prog, srcdir, opt, march, fm):
    vdir = BUILD / "variants" / name / variant_id(opt, march, fm)
    vdir.mkdir(parents=True, exist_ok=True)
    flags = [f"-{opt}", f"-march={march}", "-mabi=lp64d", "-w"]
    if fm:
        flags.append("-ffast-math")
    objs = []
    for src in prog["sources"]:
        stem = src.replace("/", "_").removesuffix(".c")
        asm, obj = vdir / f"{stem}.s", vdir / f"{stem}.o"
        run([CC, *flags, "-S", str(srcdir / src), "-o", str(asm)])
        run([CC, *flags, "-c", str(asm), "-o", str(obj)])
        objs.append(obj)
    run([CC, *flags, *map(str, objs), "-o", str(vdir / "bin"), "-lm"])
    return vdir, objs


LABEL_RE = re.compile(r"^[A-Za-z0-9_.$]+:")
FUNC_RE = re.compile(r"^\s*\.type\s+([^,\s]+)\s*,\s*@function")
OBJ_RE = re.compile(r"^\s*[0-9a-f]+:\s+([0-9a-f]+)\s+(\S+)")


def static_asm(asm_files):
    """Mnemonics in GCC's .s output (pseudo-instructions as written) + defined functions."""
    counts, funcs = {}, set()
    for path in asm_files:
        for line in path.read_text().splitlines():
            if m := FUNC_RE.match(line):
                funcs.add(m.group(1))
            line = line.split("#", 1)[0].strip()
            while m := LABEL_RE.match(line):
                line = line[m.end():].strip()
            if not line or line.startswith("."):
                continue
            mn = line.split()[0]
            counts[mn] = counts.get(mn, 0) + 1
    return counts, funcs


def static_obj(obj_files):
    """Mnemonics in the assembled objects; 2-byte encodings are tagged "c."."""
    counts = {}
    for path in obj_files:
        out = run([OBJDUMP, "-d", str(path)]).stdout
        for line in out.splitlines():
            if m := OBJ_RE.match(line):
                hexenc, mn = m.groups()
                if len(hexenc) == 4:
                    mn = "c." + mn
                counts[mn] = counts.get(mn, 0) + 1
    return counts


def run_dynamic(vdir, prog, k):
    outdir = vdir / "dyn"
    outdir.mkdir(exist_ok=True)
    out = outdir / f"tc{k}.csv"
    if out.exists():
        return out
    tmp = out.with_suffix(".tmp")
    cmd = [str(QEMU), "-L", SYSROOT, "-cpu", "max",
           "-plugin", f"{PLUGIN},outfile={tmp}", str(vdir / "bin"),
           *map(str, prog["args"](k))]
    r = run(cmd, cwd=prog["rundir"])
    (outdir / f"tc{k}.log").write_text(r.stdout)
    tmp.rename(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--testcases", type=int, default=5)
    ap.add_argument("--program", action="append", choices=list(PROGRAMS))
    ap.add_argument("--quick", action="store_true",
                    help="small matrix (O0/O2/O3 x rv64gc/rv64gcv, no fast-math) for smoke tests")
    args = ap.parse_args()

    for p in (QEMU, PLUGIN):
        if not p.exists():
            sys.exit(f"missing {p}: run tools/build_qemu.sh and make -C tools")
    if not (BUILD / "data/magicalbat/train_images.mat").exists():
        run([sys.executable, "-I", str(ROOT / "tools/make_mat.py"),
             str(ROOT / "CNN/AxelMontlahuc/MNIST"), str(BUILD / "data/magicalbat")])

    programs = args.program or list(PROGRAMS)
    opts, marches, fms = OPTS, MARCHES, FASTMATH
    if args.quick:
        opts, marches, fms = ["O0", "O2", "O3"], ["rv64gc", "rv64gcv"], [False]

    static_rows, variants = [], []
    for name in programs:
        prog = PROGRAMS[name]
        srcdir = prepare_sources(name, prog)
        for opt, march, fm in itertools.product(opts, marches, fms):
            vid = variant_id(opt, march, fm)
            print(f"build {name} {vid}", flush=True)
            vdir, objs = build_variant(name, prog, srcdir, opt, march, fm)
            asm_counts, funcs = static_asm(sorted(vdir.glob("*.s")))
            obj_counts = static_obj(objs)
            meta = dict(program=name, opt=opt, march=march, fastmath=fm, variant=vid)
            for source, counts in (("asm", asm_counts), ("obj", obj_counts)):
                static_rows += [dict(meta, source=source, mnemonic=mn, count=c)
                                for mn, c in counts.items()]
            variants.append((meta, vdir, prog, funcs))

    RESULTS.mkdir(exist_ok=True)
    static = annotate(pd.DataFrame(static_rows))
    static.to_csv(RESULTS / "static_counts.csv", index=False)

    jobs = [(v, k) for v in variants for k in range(args.testcases)]
    print(f"running {len(jobs)} QEMU runs on {args.jobs} workers", flush=True)

    def work(job):
        (meta, vdir, prog, funcs), k = job
        path = run_dynamic(vdir, prog, k)
        print(f"  done {meta['program']} {meta['variant']} tc{k}", flush=True)
        df = pd.read_csv(path, keep_default_na=False)
        df["region"] = df["symbol"].map(lambda s: "app" if s in funcs else "lib")
        for col, val in meta.items():
            df[col] = val
        df["testcase"] = k
        return df

    with ThreadPoolExecutor(args.jobs) as pool:
        raw = pd.concat(list(pool.map(work, jobs)), ignore_index=True)
    raw = annotate(raw)[VARIANT_COLS + ["testcase", "mnemonic", "base_mnemonic",
                                        "canonical", "category", "symbol",
                                        "region", "count"]]
    raw.to_csv(RESULTS / "dynamic_raw.csv", index=False)

    # Per test case totals per (variant, region, mnemonic), then mean/std over test cases.
    keys = VARIANT_COLS + ["mnemonic"]
    per_tc = pd.concat([
        raw.groupby(keys + ["region", "testcase"], as_index=False)["count"].sum(),
        raw.groupby(keys + ["testcase"], as_index=False)["count"].sum().assign(region="all"),
    ])
    wide = per_tc.pivot_table(index=keys + ["region"], columns="testcase",
                              values="count", fill_value=0)
    dyn = pd.DataFrame({"mean": wide.mean(axis=1), "std": wide.std(axis=1, ddof=0)}).reset_index()
    dyn["fraction"] = dyn["mean"] / dyn.groupby(VARIANT_COLS + ["region"])["mean"].transform("sum")
    dyn = annotate(dyn)
    dyn.to_csv(RESULTS / "dynamic_counts.csv", index=False)

    # One row per variant.
    totals = per_tc.groupby(VARIANT_COLS + ["region", "testcase"])["count"].sum()
    totals = totals.unstack("region").fillna(0)
    agg = totals.groupby(level=VARIANT_COLS).agg(["mean", "std"])
    summary = pd.DataFrame({
        "dyn_total_mean": agg[("all", "mean")],
        "dyn_total_std": agg[("all", "std")],
        "dyn_app_mean": agg[("app", "mean")],
        "dyn_lib_mean": agg[("lib", "mean")],
    })
    summary["app_share"] = summary["dyn_app_mean"] / summary["dyn_total_mean"]
    for source in ("asm", "obj"):
        summary[f"static_{source}_total"] = (
            static[static.source == source].groupby(VARIANT_COLS)["count"].sum())
    cat = (dyn[dyn.region == "app"].groupby(VARIANT_COLS + ["category"])["fraction"].sum()
           .unstack("category").fillna(0))
    cat = cat[[c for c in CATEGORY_ORDER if c in cat]].add_prefix("app_share_")
    summary = summary.join(cat).reset_index()
    summary.to_csv(RESULTS / "summary.csv", index=False)
    print(summary[["program", "variant", "static_obj_total", "dyn_total_mean",
                   "app_share"]].to_string(index=False))


if __name__ == "__main__":
    main()
