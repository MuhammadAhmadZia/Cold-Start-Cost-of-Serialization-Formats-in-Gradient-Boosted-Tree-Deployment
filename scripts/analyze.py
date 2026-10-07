"""analyze.py - turn the raw measurement records into the paper's tables.

Reads every raw_main.jsonl in the results tree, which means the flat file for
Environment A and results/env_B/raw_main.jsonl for Environment B. Writes the
summary table and one CSV per numbered table in the paper, plus a plain text
report of the headline numbers.

Time to first prediction is the sum of five stages: interpreter start-up,
the NumPy import, the model library import, artifact deserialization and the
first prediction. Every stage is timed inside a single freshly launched
process, so nothing is hidden by an interpreter that has already warmed up.

Run:  python scripts/analyze.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

STAGES = ["interpreter_floor_s", "t_numpy_s", "t_import_s",
          "t_load_s", "t_first_pred_s"]
INLIB = ["pickle", "joblib", "native"]
LIBS = ["lgb", "xgb", "cat"]
LIBNAME = {"lgb": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost"}
FMTNAME = {"pickle": "pickle", "joblib": "joblib",
           "native": "native", "onnx": "ONNX"}

report = []


def say(s=""):
    print(s)
    report.append(s)


# ---------------------------------------------------------------------------
def load():
    """Every main-experiment record, from both environments."""
    paths = [os.path.join(RES, "raw_main.jsonl")]
    paths += sorted(glob.glob(os.path.join(RES, "env_*", "raw_main.jsonl")))

    frames = []
    for p in paths:
        if not os.path.exists(p):
            continue
        d = pd.read_json(p, lines=True)
        if "env_label" not in d.columns:
            # Fall back to the folder name for data written before the label
            # was recorded in the rows themselves.
            d["env_label"] = os.path.basename(os.path.dirname(p))[4:] or "A"
        frames.append(d)

    if not frames:
        raise SystemExit("no raw_main.jsonl found under results/")

    df = pd.concat(frames, ignore_index=True)
    df = df[df.n_trees != 50]                      # drop any smoke-test rows
    df = df.drop_duplicates(
        subset=["env_label", "loader", "artifact", "batch", "repeat"])
    df["ttfp_s"] = df[STAGES].sum(axis=1)
    return df


def stage_medians(sub):
    """Median of each stage, then their sum. Shares computed from that sum so
    the parts always add to the whole."""
    med = {c: sub[c].median() for c in STAGES}
    total = sum(med.values())
    return med, total


# ---------------------------------------------------------------------------
def table_summary(df):
    keys = ["env_label", "dataset", "lib", "fmt", "n_trees", "batch"]
    metrics = ["bytes"] + STAGES + ["ttfp_s", "t_steady_p50_s",
                                    "t_steady_p95_s", "t_process_wall_s"]
    g = df.groupby(keys)[metrics].median().reset_index()
    g["n_obs"] = df.groupby(keys).size().values
    g["import_share_pct"] = g.t_import_s / g.ttfp_s * 100
    g.to_csv(os.path.join(RES, "summary.csv"), index=False)
    return g


def table3_composition(df, env="A"):
    """Composition of time to first prediction at batch size one."""
    b1 = df[(df.batch == 1) & (df.env_label == env)]
    rows = []
    for lib in LIBS:
        for fmt in INLIB + ["onnx"]:
            sub = b1[(b1.lib == lib) & (b1.fmt == fmt)]
            if sub.empty:
                continue
            med, total = stage_medians(sub)
            rows.append({
                "library": LIBNAME[lib], "format": FMTNAME[fmt],
                "interpreter_ms": round(med["interpreter_floor_s"] * 1000, 1),
                "numpy_ms": round(med["t_numpy_s"] * 1000, 1),
                "import_ms": round(med["t_import_s"] * 1000, 1),
                "load_ms": round(med["t_load_s"] * 1000, 1),
                "first_pred_ms": round(med["t_first_pred_s"] * 1000, 1),
                "total_ms": round(total * 1000, 1),
                "import_share_pct": round(med["t_import_s"] / total * 100, 1),
            })
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, "table3_composition.csv"), index=False)
    return t


def table4_variance(df):
    """Share of variance in time to first prediction explained by each factor
    on its own. Separate single-predictor fits, so the columns are not a
    decomposition and do not sum to the joint model."""
    import statsmodels.formula.api as smf

    b1 = df[(df.batch == 1) & (df.env_label == "A")].copy()
    b1["log_bytes"] = np.log10(b1.bytes.clip(lower=1))
    b1["ttfp_ms"] = b1.ttfp_s * 1000

    rows = []
    for label, group in [("In-library", b1[b1.fmt.isin(INLIB)]),
                         ("ONNX", b1[b1.fmt == "onnx"])]:
        res = {}
        res["Library"] = smf.ols("ttfp_ms ~ C(lib)", group).fit().rsquared
        res["Log artifact size"] = smf.ols("ttfp_ms ~ log_bytes",
                                           group).fit().rsquared
        if group.fmt.nunique() > 1:
            res["Serialization format"] = smf.ols("ttfp_ms ~ C(fmt)",
                                                  group).fit().rsquared
        else:
            res["Serialization format"] = np.nan
        res["Library and log size together"] = smf.ols(
            "ttfp_ms ~ C(lib) + log_bytes", group).fit().rsquared
        rows.append(pd.Series(res, name=label))

    t = pd.DataFrame(rows).T.round(3)
    t.index.name = "predictor"
    t.to_csv(os.path.join(RES, "table4_variance.csv"))
    return t


def table5_equivalence(df, margins=(87.0, 50.0)):
    """Two one-sided tests between the in-library formats at batch size one.

    The unit of analysis is the experimental cell, so each difference compares
    two formats measured on the same library, dataset and model size, with the
    repeats inside a cell reduced to their median first."""
    from scipy import stats

    b1 = df[(df.batch == 1) & (df.env_label == "A") & df.fmt.isin(INLIB)]
    key = ["lib", "dataset", "n_trees"]
    wide = b1.pivot_table(index=key, columns="fmt",
                          values="ttfp_s", aggfunc="median") * 1000

    rows = []
    for a, b in [("pickle", "joblib"), ("pickle", "native"),
                 ("joblib", "native")]:
        d = (wide[a] - wide[b]).dropna()
        n = len(d)
        mean, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
        row = {"comparison": f"{a} against {b}",
               "n_pairs": n,
               "mean_difference_ms": round(mean, 1),
               "standard_error": round(se, 1)}
        for m in margins:
            t_lo = (mean + m) / se
            t_hi = (mean - m) / se
            p = max(stats.t.sf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1))
            row[f"p_margin_{int(m)}ms"] = round(p, 4)
        rows.append(row)

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, "table5_equivalence.csv"), index=False)
    return t


def table7_environments():
    """The recorded properties of each measurement environment."""
    out = {}
    for env in ("A", "B"):
        p = os.path.join(RES, f"env_{env}", "environment.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        pk = j.get("packages", {})
        out[f"Environment {env}"] = {
            "Processor": j["cpu"]["model"] or "not recorded",
            "Logical cores": j["cpu"]["logical_cores"] or "not recorded",
            "Memory GB": j["memory"]["total_gb"] or "not recorded",
            "Kernel": j["kernel"],
            "C library": j["platform"].split("with-")[-1],
            "Python": j["python"],
            "LightGBM": pk.get("lightgbm"),
            "XGBoost": pk.get("xgboost"),
            "CatBoost": pk.get("catboost"),
            "ONNX Runtime": pk.get("onnxruntime"),
        }
    t = pd.DataFrame(out)
    t.index.name = "property"
    t.to_csv(os.path.join(RES, "table7_environments.csv"))
    return t


def table8_replication(df):
    """The headline numbers side by side for both environments."""
    out = {}
    for env in sorted(df.env_label.unique()):
        b1 = df[(df.batch == 1) & (df.env_label == env)]
        col = {}
        for lib in LIBS:
            # One share per serialization format, computed exactly as in the
            # composition table, then the median of the three. Keeping the
            # two tables on the same convention is what lets them be read
            # against each other.
            shares = []
            for fmt in INLIB:
                sub = b1[(b1.lib == lib) & (b1.fmt == fmt)]
                med, total = stage_medians(sub)
                shares.append(round(med["t_import_s"] / total * 100, 1))
            col[f"Import share, {LIBNAME[lib]}"] = float(np.median(shares))
        for fmt in INLIB + ["onnx"]:
            sub = b1[b1.fmt == fmt]
            col[f"Median TTFP, {FMTNAME[fmt]} (ms)"] = \
                round(sub.ttfp_s.median() * 1000, 0)
        nat = b1[b1.fmt == "native"].ttfp_s.median()
        onx = b1[b1.fmt == "onnx"].ttfp_s.median()
        col["ONNX advantage at batch one"] = round(nat / onx, 2)
        out[f"Environment {env}"] = col

    t = pd.DataFrame(out)
    t.index.name = "measure"
    t.to_csv(os.path.join(RES, "table8_replication.csv"))
    return t


# ---------------------------------------------------------------------------
def headline(df):
    say("=" * 70)
    say("COLD START ANALYSIS")
    say("=" * 70)
    for env in sorted(df.env_label.unique()):
        say(f"  Environment {env}: {len(df[df.env_label == env])} "
            f"main-experiment measurements")
    say()

    b1 = df[df.batch == 1]
    say("Import share of time to first prediction, in-library formats")
    lo_hi = {}
    for env in sorted(df.env_label.unique()):
        shares = []
        for lib in LIBS:
            for fmt in INLIB:
                sub = b1[(b1.env_label == env) & (b1.lib == lib)
                         & (b1.fmt == fmt)]
                med, total = stage_medians(sub)
                shares.append(med["t_import_s"] / total * 100)
        lo_hi[env] = (min(shares), max(shares))
        say(f"   Environment {env}: {min(shares):.1f} to {max(shares):.1f} "
            f"percent across the nine library and format pairs")
    allshares = [v for pair in lo_hi.values() for v in pair]
    say(f"   Across both environments: {min(allshares):.1f} to "
        f"{max(allshares):.1f} percent")
    say()

    say("Deserialization share of time to first prediction, in-library formats")
    for env in sorted(df.env_label.unique()):
        sub = b1[(b1.env_label == env) & b1.fmt.isin(INLIB)]
        med, total = stage_medians(sub)
        say(f"   Environment {env}: {med['t_load_s'] / total * 100:.2f} percent")
    say()

    say("Spread of time to first prediction and of artifact size")
    for env in sorted(df.env_label.unique()):
        sub = b1[(b1.env_label == env) & b1.fmt.isin(INLIB)]
        cells = sub.groupby(["lib", "fmt", "dataset", "n_trees"]).ttfp_s.median()
        allsz = b1[b1.env_label == env]
        szs = allsz[allsz.bytes > 0].bytes
        say(f"   Environment {env}: time varies by {cells.max()/cells.min():.2f} "
            f"times, artifact size by {szs.max()/szs.min():.0f} times")
    say()

    say("ONNX advantage on time to first prediction at batch size one")
    for env in sorted(df.env_label.unique()):
        sub = b1[b1.env_label == env]
        nat = sub[sub.fmt == "native"].ttfp_s.median()
        onx = sub[sub.fmt == "onnx"].ttfp_s.median()
        say(f"   Environment {env}: {nat/onx:.2f} times faster "
            f"({nat*1000:.0f} ms against {onx*1000:.0f} ms)")
    say()


def figures(df):
    """The four analysis figures. Figure numbering here follows the order the
    figures are produced, not the order they appear in the paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    palette = {"pickle": "#8c7ae6", "joblib": "#487eb0",
               "native": "#e1b12c", "onnx": "#44bd32"}
    order = INLIB + ["onnx"]
    b1 = df[(df.batch == 1) & (df.env_label == "A")]

    # 1. Where time to first prediction goes.
    idx, cols = [], []
    for lib in LIBS:
        for fmt in order:
            sub = b1[(b1.lib == lib) & (b1.fmt == fmt)]
            if sub.empty:
                continue
            med, _ = stage_medians(sub)
            idx.append(f"{LIBNAME[lib]}\n{FMTNAME[fmt]}")
            cols.append([med[c] * 1000 for c in STAGES])
    arr = np.array(cols)

    parts = [("interpreter", "#dcdde1"), ("NumPy", "#b2bec3"),
             ("library import", "#e17055"), ("artifact load", "#0984e3"),
             ("first prediction", "#00b894")]
    fig, ax = plt.subplots(figsize=(13.6, 4.6))
    bottom = np.zeros(len(idx))
    for i, (name, colour) in enumerate(parts):
        ax.bar(idx, arr[:, i], bottom=bottom, label=name, color=colour,
               edgecolor="white", linewidth=0.6)
        bottom += arr[:, i]
    ax.set_ylabel("milliseconds")
    ax.legend(frameon=False, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.14))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_breakdown.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    # 2. Artifact size against cold start cost.
    g = b1[b1.bytes > 0].groupby(["fmt", "lib", "dataset", "n_trees"]).agg(
        bytes=("bytes", "median"), ttfp=("ttfp_s", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for fmt in order:
        s = g[g.fmt == fmt]
        if s.empty:
            continue
        ax.scatter(s.bytes / 1e6, s.ttfp * 1000, label=FMTNAME[fmt], s=46,
                   color=palette[fmt], alpha=0.85, edgecolor="white")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("artifact size in megabytes")
    ax.set_ylabel("time to first prediction in milliseconds")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_size_vs_cold.png"), dpi=200)
    plt.close(fig)

    # 3. Cold start against steady state, per library.
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
    for ax, lib in zip(axes, LIBS):
        sub = b1[b1.lib == lib]
        cold = sub.groupby("fmt").ttfp_s.median() * 1000
        warm = sub.groupby("fmt").t_steady_p50_s.median() * 1000
        x = np.arange(len(order))
        ax.bar(x - 0.2, [cold.get(f, np.nan) for f in order], 0.4,
               label="cold start", color="#e17055")
        ax.bar(x + 0.2, [warm.get(f, np.nan) for f in order], 0.4,
               label="steady state", color="#00b894")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([FMTNAME[f] for f in order], fontsize=9)
        ax.set_title(LIBNAME[lib], fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.22, axis="y", which="both")
    axes[0].set_ylabel("milliseconds, log scale")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_flip_per_lib.png"), dpi=200)
    plt.close(fig)

    # 4. The ONNX advantage on cold start, per library and dataset.
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    width, x = 0.26, np.arange(3)
    for i, ds in enumerate(["breast", "calif", "covtype"]):
        vals = []
        for lib in LIBS:
            s = b1[(b1.lib == lib) & (b1.dataset == ds)]
            nat = s[s.fmt == "native"].ttfp_s.median()
            onx = s[s.fmt == "onnx"].ttfp_s.median()
            vals.append(nat / onx if onx and not np.isnan(onx) else np.nan)
        ax.bar(x + (i - 1) * width, vals, width,
               label={"breast": "Breast Cancer", "calif": "California Housing",
                      "covtype": "Covertype"}[ds],
               color=["#8c7ae6", "#487eb0", "#44bd32"][i])
    ax.axhline(1.0, color="#2d3436", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([LIBNAME[l] for l in LIBS])
    ax.set_ylabel("native divided by ONNX")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.22, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_onnx_advantage.png"), dpi=200)
    plt.close(fig)

    print(f"figures -> {FIG}/")


def main():
    df = load()
    table_summary(df)
    headline(df)

    say("-" * 70)
    say("Table 3  Composition of time to first prediction, Environment A")
    say(table3_composition(df).to_string(index=False))
    say()

    say("-" * 70)
    say("Table 4  Variance explained by each factor alone")
    say(table4_variance(df).to_string())
    say()

    say("-" * 70)
    say("Table 5  Equivalence tests among the in-library formats")
    say(table5_equivalence(df).to_string(index=False))
    say()

    say("-" * 70)
    say("Table 7  Measurement environments")
    say(table7_environments().to_string())
    say()

    say("-" * 70)
    say("Table 8  Replication across environments")
    say(table8_replication(df).to_string())
    say()
    figures(df)
    say()
    say("Table 6, the crossover batch sizes, is produced by "
        "scripts/crossover_analysis.py.")

    with open(os.path.join(RES, "analysis_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"\nreport -> {os.path.join(RES, 'analysis_report.txt')}")


if __name__ == "__main__":
    main()
