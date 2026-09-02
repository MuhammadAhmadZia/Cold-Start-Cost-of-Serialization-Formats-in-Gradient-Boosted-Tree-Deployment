"""
analyze.py - regenerate every table and figure reported in the paper.

Reads results/raw_main.jsonl and results/manifest.csv. Writes the summary
tables to results/ and the figures to figures/. Running this on the shipped
measurement records reproduces the published numbers without repeating the
benchmark.

Run:  python scripts/analyze.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

PALETTE = {"pickle": "#8c7ae6", "joblib": "#487eb0",
           "native": "#e1b12c", "onnx": "#44bd32"}
ORDER = ["pickle", "joblib", "native", "onnx"]
LIBNAME = {"lgb": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost"}

report = []


def say(line=""):
    print(line)
    report.append(line)


# ---------------------------------------------------------------------------
df = pd.read_json(os.path.join(RES, "raw_main.jsonl"), lines=True)

# Time to first prediction as a user experiences it: process launch to answer.
df["ttfp"] = (df.interpreter_floor_s + df.t_numpy_s + df.t_import_s
              + df.t_load_s + df.t_first_pred_s) * 1000
df["import_share"] = df.t_import_s * 1000 / df.ttfp * 100
df["load_share"] = df.t_load_s * 1000 / df.ttfp * 100
df["MB"] = df.bytes / 1e6
b1 = df[df.batch == 1]

keys = ["dataset", "lib", "fmt", "n_trees", "batch"]
mets = ["bytes", "t_numpy_s", "t_import_s", "t_load_s", "t_first_pred_s",
        "t_steady_p50_s", "t_steady_p95_s", "ttfp", "import_share"]
summary = df.groupby(keys)[mets].median().reset_index()
summary["n_obs"] = df.groupby(keys).size().values
summary.to_csv(os.path.join(RES, "summary.csv"), index=False)

say(f"measurements: {len(df)}   cells: {df.groupby(keys).ngroups}")
say(f"interpreter floor: {df.interpreter_floor_s.median()*1000:.1f} ms   "
    f"numpy import: {df.t_numpy_s.median()*1000:.1f} ms")
say()

# --- Table 3: composition of the cold start tax -----------------------------
say("TABLE 3  Composition of time to first prediction at batch 1 (ms)")
t3 = b1.groupby(["lib", "fmt"])[["t_import_s", "t_load_s", "t_first_pred_s",
                                 "ttfp", "import_share"]].median()
t3[["t_import_s", "t_load_s", "t_first_pred_s"]] *= 1000
t3.columns = ["import", "load", "first_pred", "ttfp", "import_pct"]
say(t3.round(1).to_string())
t3.round(2).to_csv(os.path.join(RES, "table3_composition.csv"))
tree_share = b1[b1.fmt != "onnx"].groupby(["lib", "fmt"]).import_share.median()
say(f"\nimport share for in-library formats: "
    f"{tree_share.min():.1f}% to {tree_share.max():.1f}%")
say(f"median load share: {b1[b1.fmt!='onnx'].load_share.median():.1f}%")
ttfp_fmt = b1.groupby("fmt").ttfp.median()
say(f"ONNX speedup vs native on TTFP: "
    f"{ttfp_fmt['native']/ttfp_fmt['onnx']:.2f}x")
say()

# --- Table 4: variance decomposition ----------------------------------------
say("TABLE 4  Share of variance in TTFP explained by each factor alone")
tree, onnx = b1[b1.fmt != "onnx"], b1[b1.fmt == "onnx"]
rows = []
for label, sub in [("in-library", tree), ("onnx", onnx)]:
    r = {"subset": label}
    r["library"] = smf.ols("ttfp ~ C(lib)", data=sub).fit().rsquared
    r["log_size"] = smf.ols("ttfp ~ np.log10(MB)", data=sub).fit().rsquared
    r["format"] = (smf.ols("ttfp ~ C(fmt)", data=sub).fit().rsquared
                   if label == "in-library" else np.nan)
    r["library_and_size"] = smf.ols("ttfp ~ C(lib)+np.log10(MB)",
                                    data=sub).fit().rsquared
    rows.append(r)
t4 = pd.DataFrame(rows).set_index("subset")
say(t4.round(3).to_string())
t4.round(4).to_csv(os.path.join(RES, "table4_variance.csv"))

for label, sub in [("in-library", tree), ("onnx", onnx)]:
    s = sub.copy()
    s["load_ms"] = s.t_load_s * 1000
    fit = smf.ols("load_ms ~ MB", data=s).fit()
    say(f"  {label:11s} load time on size: R2={fit.rsquared:.3f}, "
        f"{fit.params['MB']:.1f} ms/MB, load is "
        f"{s.load_share.median():.1f}% of TTFP")
sizes = df.bytes
say(f"  artifact size spans {sizes.min()/1e6:.3f} to {sizes.max()/1e6:.2f} MB "
    f"({sizes.max()/sizes.min():.0f}x)")
cellttfp = tree.groupby(["dataset", "lib", "n_trees", "fmt"]).ttfp.median()
say(f"  TTFP for in-library formats spans "
    f"{cellttfp.max()/cellttfp.min():.2f}x")
say()

# --- Table 5: equivalence tests ---------------------------------------------
say("TABLE 5  Paired equivalence tests among the in-library formats")
piv = b1.pivot_table(index=["dataset", "lib", "n_trees"], columns="fmt",
                     values="ttfp", aggfunc="median")
margin = 0.05 * piv[["pickle", "joblib", "native"]].values.mean()
say(f"equivalence margin: 5% of mean TTFP = {margin:.0f} ms "
    f"over {len(piv)} paired configurations")


def tost(a, b, m):
    d = a - b
    n = len(d)
    mu = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    return mu, se, max(1 - stats.t.cdf((mu + m) / se, n - 1),
                       stats.t.cdf((mu - m) / se, n - 1))


rows = []
for x, y in [("pickle", "joblib"), ("pickle", "native"), ("joblib", "native")]:
    mu, se, p_pre = tost(piv[x].values, piv[y].values, margin)
    _, _, p_str = tost(piv[x].values, piv[y].values, 50)
    rows.append(dict(comparison=f"{x} vs {y}", mean_diff_ms=round(mu, 1),
                     std_error=round(se, 1), p_margin_5pct=round(p_pre, 4),
                     p_margin_50ms=round(p_str, 4)))
t5 = pd.DataFrame(rows)
say(t5.to_string(index=False))
t5.to_csv(os.path.join(RES, "table5_equivalence.csv"), index=False)
say()

# --- Figure: composition ----------------------------------------------------
g = b1.groupby(["lib", "fmt"])[["interpreter_floor_s", "t_numpy_s",
                                "t_import_s", "t_load_s",
                                "t_first_pred_s"]].median()
g = g.reindex([(l, f) for l in ["lgb", "xgb", "cat"] for f in ORDER
               if (l, f) in g.index])
fig, ax = plt.subplots(figsize=(11, 5))
bottom = np.zeros(len(g))
for name, col, c in [("interpreter", "interpreter_floor_s", "#dcdde1"),
                     ("numpy", "t_numpy_s", "#b2bec3"),
                     ("library import", "t_import_s", "#e17055"),
                     ("model load", "t_load_s", "#0984e3"),
                     ("first predict", "t_first_pred_s", "#00b894")]:
    v = g[col].values * 1000
    ax.bar([f"{LIBNAME[l]}\n{f}" for l, f in g.index], v, bottom=bottom,
           label=name, color=c, edgecolor="white", linewidth=.6)
    bottom += v
ax.set_ylabel("milliseconds")
ax.set_title("Composition of time to first prediction at batch size one")
ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(.5, -.12))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_breakdown.png"), dpi=200,
            bbox_inches="tight")
plt.close(fig)

# --- Figure: size against cold start ----------------------------------------
g2 = b1.groupby(["fmt", "lib", "dataset", "n_trees"]).agg(
    MB=("MB", "median"), ttfp=("ttfp", "median")).reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
for fmt in ORDER:
    s = g2[g2.fmt == fmt]
    ax.scatter(s.MB, s.ttfp, label=fmt, s=52, color=PALETTE[fmt],
               alpha=.85, edgecolor="white")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("artifact size (MB, log)")
ax.set_ylabel("time to first prediction (ms, log)")
ax.set_title("Artifact size against cold start cost")
ax.legend(frameon=False, title="format"); ax.grid(alpha=.25, which="both")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_size_vs_cold.png"), dpi=200)
plt.close(fig)

# --- Figure: steady state per library ---------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, lib in zip(axes, ["lgb", "xgb", "cat"]):
    s = df[df.lib == lib]
    for fmt in ORDER:
        t = s[s.fmt == fmt].groupby("batch").t_steady_p50_s.median()
        ax.plot(t.index, t.values * 1000, marker="o", label=fmt,
                color=PALETTE[fmt], lw=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(LIBNAME[lib]); ax.set_xlabel("batch size (log)")
    ax.grid(alpha=.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("steady-state p50 latency (ms, log)")
axes[2].legend(frameon=False, title="format")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_flip_per_lib.png"), dpi=200,
            bbox_inches="tight")
plt.close(fig)

# --- Prediction agreement ---------------------------------------------------
eq_path = os.path.join(RES, "equivalence.csv")
if os.path.exists(eq_path):
    eq = pd.read_csv(eq_path)
    say(f"PREDICTION AGREEMENT  {int(eq.ok.sum())}/{len(eq)} configurations "
        f"agree; largest absolute difference {eq.max_abs_diff.max():.2e}")

open(os.path.join(RES, "analysis_report.txt"), "w").write("\n".join(report) + "\n")
print(f"\ntables written to {RES}\nfigures written to {FIG}")
