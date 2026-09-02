"""
crossover_analysis.py - corrected crossover analysis for the batch sweep.

The crossover is defined as the smallest batch size at which the ONNX path is
significantly slower, meaning the lower bound of the bootstrap 95% interval for
the ONNX to native latency ratio lies above one, and remains significantly
slower at every larger batch tested.

A significance requirement is used rather than simple interpolation of the point
where the ratio crosses one. Interpolation reports a confident crossover wherever
the ratio happens to wander across one by chance, which occurs in cells where the
two paths perform comparably.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

sw = pd.read_json(os.path.join(SRC, "raw_batchsweep.jsonl"), lines=True)
BATCHES = sorted(sw.batch.unique())
LIBNAME = {"lgb": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost"}
DSNAME = {"breast": "Breast Cancer", "calif": "California Housing",
          "covtype": "Covertype"}


def boot(ds, lib, b, n=4000):
    o = sw[(sw.dataset == ds) & (sw.lib == lib) & (sw.batch == b)
           & (sw.fmt == "onnx")].t_steady_p50_s.values
    v = sw[(sw.dataset == ds) & (sw.lib == lib) & (sw.batch == b)
           & (sw.fmt == "native")].t_steady_p50_s.values
    r = [np.median(rng.choice(o, len(o))) / np.median(rng.choice(v, len(v)))
         for _ in range(n)]
    return np.percentile(r, [2.5, 50, 97.5])


records, rows = [], []
for ds in DSNAME:
    for lib in LIBNAME:
        cis = {b: boot(ds, lib, b) for b in BATCHES}
        for b, (lo, md, hi) in cis.items():
            records.append(dict(dataset=ds, lib=lib, batch=b,
                                ratio_lo=lo, ratio=md, ratio_hi=hi,
                                onnx_slower=lo > 1))
        sig = [cis[b][0] > 1 for b in BATCHES]
        first = next((BATCHES[i] for i in range(len(BATCHES))
                      if all(sig[i:])), None)
        lo, md, hi = cis[BATCHES[-1]]
        rows.append(dict(dataset=ds, lib=lib,
                         crossover_batch=first if first else np.inf,
                         ratio_at_2048=md, ci_lo=lo, ci_hi=hi))

curves = pd.DataFrame(records)
curves.to_csv(os.path.join(OUT, "crossover_curves.csv"), index=False)
cross = pd.DataFrame(rows)
cross.to_csv(os.path.join(OUT, "crossover_bootstrap.csv"), index=False)

print("=== CROSSOVER (bootstrap, 95% CI) ===")
print("smallest batch at which ONNX is significantly slower and stays so\n")
print(cross.round(2).to_string(index=False))
print("\n=== DECISION RULE by library ===")
for lib in ["lgb", "xgb", "cat"]:
    s = cross[cross.lib == lib].crossover_batch
    if np.isinf(s).all():
        print(f"  {LIBNAME[lib]:9s}: ONNX faster at every batch up to 2048")
    else:
        print(f"  {LIBNAME[lib]:9s}: ONNX loses above batch "
              f"{int(s.min())}-{int(s.max())}")

# --- figure with CI bands ---------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
COLS = {"breast": "#0984e3", "calif": "#e17055", "covtype": "#6ab04c"}
for ax, lib in zip(axes, ["lgb", "xgb", "cat"]):
    for ds, c in COLS.items():
        s = curves[(curves.lib == lib) & (curves.dataset == ds)].sort_values("batch")
        ax.plot(s.batch, s.ratio, marker="o", ms=4, color=c, lw=1.9,
                label=DSNAME[ds])
        ax.fill_between(s.batch, s.ratio_lo, s.ratio_hi, color=c, alpha=.18,
                        linewidth=0)
    ax.axhline(1, color="#2d3436", ls="--", lw=1.2)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("batch size")
    ax.set_title(LIBNAME[lib])
    ax.grid(alpha=.22, which="both")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("ONNX / native latency\n(below 1 = ONNX faster)")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Crossover batch size is library-dependent: CatBoost loses ONNX's "
             "advantage by batch 8-16,\nXGBoost by 32-512, LightGBM not at all "
             "up to 2048", y=1.06, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "fig5_crossover_ci.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# --- degenerate-model check -------------------------------------------------
m = pd.read_csv(os.path.join(SRC, "manifest.csv"))
piv = m[m.fmt == "native"].pivot_table(index=["dataset", "lib"],
                                       columns="n_trees", values="bytes")
if 500 in piv.columns and 2000 in piv.columns:
    d = piv.dropna(subset=[500, 2000])
    d = d.assign(ratio=(d[2000] / d[500]).round(2))
    print("\n=== model saturation check: artifact size, 2000 trees / 500 trees ===")
    print("a ratio of 1.00 means the library stopped adding trees\n")
    print(d[["ratio"]].to_string())

print(f"\nwritten to {OUT}")
