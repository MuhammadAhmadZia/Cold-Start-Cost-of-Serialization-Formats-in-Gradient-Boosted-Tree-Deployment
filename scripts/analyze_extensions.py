"""
analyze_extensions.py - analysis for the two follow-up experiments.

Part A, cross-environment: compares the main benchmark across every
results/env_*/raw_main.jsonl that exists, and reports whether the import-share
finding and the crossover batch sizes replicate.

Part B, cold cache: compares eviction levels within an environment and reports
how much the warm page cache flattered each format.

Both parts skip themselves if their data is not present, so this can be run at
any point during data collection.

Run:  python scripts/analyze_extensions.py
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

LIBNAME = {"lgb": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost"}
PALETTE = {"pickle": "#8c7ae6", "joblib": "#487eb0",
           "native": "#e1b12c", "onnx": "#44bd32"}

report = []


def say(s=""):
    print(s)
    report.append(s)


def ttfp(df):
    return (df.interpreter_floor_s + df.t_numpy_s + df.t_import_s
            + df.t_load_s + df.t_first_pred_s) * 1000


# ===========================================================================
# Part A: cross-environment replication
# ===========================================================================
def part_a():
    frames = []
    for path in sorted(glob.glob(os.path.join(RES, "env_*", "raw_main.jsonl"))):
        label = os.path.basename(os.path.dirname(path))[4:]
        d = pd.read_json(path, lines=True)
        d["env"] = label
        frames.append(d)

    base = os.path.join(RES, "raw_main.jsonl")
    if os.path.exists(base):
        d = pd.read_json(base, lines=True)
        d["env"] = d.get("env_label", pd.Series(["original"] * len(d)))
        d["env"] = d["env"].fillna("original")
        frames.append(d)

    if not frames:
        say("Part A skipped: no raw_main.jsonl found in any environment.")
        return None

    df = pd.concat(frames, ignore_index=True)
    df = df[df.n_trees != 50]
    df["ttfp"] = ttfp(df)
    df["import_share"] = df.t_import_s * 1000 / df.ttfp * 100

    envs = sorted(df.env.unique())
    say("=" * 66)
    say("PART A  CROSS-ENVIRONMENT REPLICATION")
    say("=" * 66)
    say(f"environments: {', '.join(envs)}")
    for e in envs:
        say(f"   {e:12s} {len(df[df.env==e]):5d} measurements")
    say()

    for e in envs:
        p = os.path.join(RES, f"env_{e}", "environment.json")
        if os.path.exists(p):
            j = json.load(open(p))
            say(f"   {e:12s} {j['cpu']['model']}, "
                f"{j['cpu']['logical_cores']} cores, "
                f"{j['memory']['total_gb']} GB, Python {j['python']}")
    say()

    b1 = df[df.batch == 1]
    say("Import share of time to first prediction, in-library formats (%)")
    # One share per serialization format, from stage medians so the five
    # stages always add to the whole, then the median of the three. Same
    # convention as the composition and replication tables.
    stages = ["interpreter_floor_s", "t_numpy_s", "t_import_s",
              "t_load_s", "t_first_pred_s"]
    med = b1[b1.fmt != "onnx"].groupby(["env", "lib", "fmt"])[stages].median()
    per_fmt = (med.t_import_s / med.sum(axis=1) * 100).round(1)
    t = per_fmt.groupby(level=["env", "lib"]).median().unstack()
    say(t.round(1).to_string())
    say()

    say("Median time to first prediction by format and environment (ms)")
    t2 = b1.groupby(["env", "fmt"]).ttfp.median().unstack()
    say(t2.round(0).to_string())
    if "native" in t2.columns and "onnx" in t2.columns:
        say()
        say("ONNX speedup on time to first prediction")
        for e in t2.index:
            say(f"   {e:12s} {t2.loc[e,'native']/t2.loc[e,'onnx']:.2f}x")
    say()

    if len(envs) > 1:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        w = 0.8 / len(envs)
        fmts = ["pickle", "joblib", "native", "onnx"]
        x = np.arange(len(fmts))
        for i, e in enumerate(envs):
            v = [b1[(b1.env == e) & (b1.fmt == f)].ttfp.median() for f in fmts]
            ax.bar(x + i * w, v, w, label=e, edgecolor="white")
        ax.set_xticks(x + w * (len(envs) - 1) / 2)
        ax.set_xticklabels(fmts)
        ax.set_yscale("log")
        ax.set_ylabel("time to first prediction (ms, log)")
        ax.set_title("Cold start by format across environments")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig7_environments.png"), dpi=200)
        plt.close(fig)
        say("wrote figures/fig7_environments.png")
    else:
        say("Only one environment present, so no comparison figure was drawn.")
    say()
    return df


# ===========================================================================
# Part B: cold page cache
# ===========================================================================
def part_b():
    paths = sorted(glob.glob(os.path.join(RES, "env_*", "raw_coldcache.jsonl")))
    if not paths:
        say("Part B skipped: no raw_coldcache.jsonl found.")
        return

    frames = []
    for p in paths:
        d = pd.read_json(p, lines=True)
        d["env"] = os.path.basename(os.path.dirname(p))[4:]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["wall_ms"] = df.t_process_wall_s * 1000

    order = [lv for lv in ["warm", "artifact", "libs", "full"]
             if lv in set(df.eviction)]

    say("=" * 66)
    say("PART B  COLD PAGE CACHE")
    say("=" * 66)
    say(f"environments: {', '.join(sorted(df.env.unique()))}")
    say(f"eviction levels: {', '.join(order)}   "
        f"{len(df)} measurements\n")

    say("Median total process wall time by eviction level (ms)")
    t = df.groupby(["env", "eviction", "fmt"]).wall_ms.median().unstack()
    t = t.reindex([(e, lv) for e in sorted(df.env.unique()) for lv in order
                   if (e, lv) in t.index])
    say(t.round(1).to_string())
    say()

    say("Cost of a cold cache, relative to the warm-cache control arm")
    for e in sorted(df.env.unique()):
        sub = df[df.env == e]
        warm = sub[sub.eviction == "warm"].groupby("fmt").wall_ms.median()
        for lv in order:
            if lv == "warm":
                continue
            cur = sub[sub.eviction == lv].groupby("fmt").wall_ms.median()
            parts = [f"{f} {cur[f]/warm[f]:.2f}x" for f in cur.index
                     if f in warm.index]
            say(f"   {e:10s} {lv:9s} {', '.join(parts)}")
    say()

    # Does a cold cache narrow the ONNX advantage?
    say("ONNX speedup over native, by eviction level")
    for e in sorted(df.env.unique()):
        sub = df[df.env == e]
        for lv in order:
            g = sub[sub.eviction == lv].groupby("fmt").wall_ms.median()
            if {"native", "onnx"} <= set(g.index):
                say(f"   {e:10s} {lv:9s} {g['native']/g['onnx']:.2f}x")
    say()

    fig, ax = plt.subplots(figsize=(8, 4.6))
    fmts = [f for f in ["native", "onnx"] if f in set(df.fmt)]
    x = np.arange(len(order))
    w = 0.8 / max(len(fmts), 1)
    for i, f in enumerate(fmts):
        v = [df[(df.eviction == lv) & (df.fmt == f)].wall_ms.median()
             for lv in order]
        ax.bar(x + i * w, v, w, label=f, color=PALETTE.get(f), edgecolor="white")
    ax.set_xticks(x + w * (len(fmts) - 1) / 2)
    ax.set_xticklabels(order)
    ax.set_ylabel("total process wall time (ms)")
    ax.set_xlabel("page cache eviction level")
    ax.set_title("Cold start under progressively colder page cache")
    ax.legend(frameon=False, title="format")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_coldcache.png"), dpi=200)
    plt.close(fig)
    say("wrote figures/fig8_coldcache.png")


if __name__ == "__main__":
    part_a()
    say()
    part_b()
    out = os.path.join(RES, "extensions_report.txt")
    open(out, "w").write("\n".join(report) + "\n")
    print(f"\nreport written to {out}")
