"""
make_method_figures.py - draws the two methodology diagrams as PNGs.

No AI image generation. Everything here is matplotlib primitives, so every box,
arrow and label can be moved or relabelled by editing the coordinates below.

Figure 1: the overall experimental pipeline (abstract view).
Figure 2: the cold-start measurement protocol inside one probe process.

Run:  python make_method_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import os
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# Muted palette that survives greyscale printing.
C_STAGE = "#e8eef5"      # stage background
C_BOX = "#ffffff"        # process box
C_ACCENT = "#2d6cb5"     # borders and arrows
C_WARM = "#c0653a"       # highlighted / isolated element
C_TEXT = "#1c2833"
FONT = 9


def box(ax, x, y, w, h, text, fc=C_BOX, ec=C_ACCENT, fs=FONT, lw=1.2,
        weight="normal", style="round,pad=0.012"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=C_TEXT, weight=weight, linespacing=1.45)


def arrow(ax, p1, p2, style="-|>", lw=1.3, color=C_ACCENT, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


def band(ax, x, y, w, h, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                                facecolor=C_STAGE, edgecolor="#b8c6d6",
                                linewidth=1.0))
    ax.text(x + 0.012, y + h - 0.03, label, ha="left", va="top",
            fontsize=FONT + 0.5, weight="bold", color="#33475b")


# ---------------------------------------------------------------------------
# FIGURE 1 - overall experimental pipeline
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 6.0))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# --- Stage A: model preparation -------------------------------------------
band(ax, 0.015, 0.655, 0.97, 0.325, "Stage A   Model Preparation (run once)")

box(ax, 0.045, 0.700, 0.155, 0.175,
    "Three datasets\n\nBreast Cancer\nCalifornia Housing\nCovertype", fs=FONT - 0.5)
box(ax, 0.240, 0.700, 0.155, 0.175,
    "Three libraries\n\nLightGBM\nXGBoost\nCatBoost", fs=FONT - 0.5)
box(ax, 0.435, 0.700, 0.145, 0.175,
    "Model sizes\n\n100 / 500 / 2000\ntrees\n(fixed seed)", fs=FONT - 0.5)
box(ax, 0.620, 0.700, 0.150, 0.175,
    "Export to four\nformats\n\npickle  joblib\nnative  ONNX", fs=FONT - 0.5)
box(ax, 0.810, 0.700, 0.150, 0.175,
    "96 artifacts\n\nsize, export time\nrecorded in\nmanifest", fs=FONT - 0.5,
    fc="#eef4ec", ec="#4a7c59")

for x1, x2 in [(0.200, 0.240), (0.395, 0.435), (0.580, 0.620), (0.770, 0.810)]:
    arrow(ax, (x1, 0.7875), (x2, 0.7875))

# --- Stage B: measurement --------------------------------------------------
band(ax, 0.015, 0.235, 0.97, 0.385, "Stage B   Cold-Start Measurement")

box(ax, 0.045, 0.290, 0.180, 0.210,
    "Orchestrator\n\nloops over every\nartifact, batch\nand repeat", fs=FONT - 0.5)

box(ax, 0.290, 0.290, 0.300, 0.235,
    "", fc="#fbf1ec", ec=C_WARM, lw=1.6)
ax.text(0.440, 0.500, "Fresh operating-system process",
        ha="center", va="center", fontsize=FONT, weight="bold", color=C_WARM)
box(ax, 0.310, 0.310, 0.125, 0.150,
    "Probe\n\nloads one\nartifact,\npredicts", fs=FONT - 0.5)
box(ax, 0.455, 0.310, 0.115, 0.150,
    "Five timed\nphases\n\n(see Fig. 2)", fs=FONT - 0.5)
arrow(ax, (0.435, 0.385), (0.455, 0.385))

box(ax, 0.655, 0.300, 0.140, 0.215,
    "One JSON\nrecord\n\nper cold start", fs=FONT - 0.5)
box(ax, 0.830, 0.300, 0.140, 0.215,
    "2,520 records\n\n1,440 main run\n1,080 batch\nsweep", fs=FONT - 0.5,
    fc="#eef4ec", ec="#4a7c59")

arrow(ax, (0.225, 0.395), (0.290, 0.395))
arrow(ax, (0.590, 0.4075), (0.655, 0.4075))
arrow(ax, (0.795, 0.4075), (0.830, 0.4075))
arrow(ax, (0.885, 0.700), (0.885, 0.528), ls="--")
ax.text(0.897, 0.610, "artifacts", ha="left", va="center",
        fontsize=FONT - 1, color="#5a6b7c", rotation=90)

# --- Stage C: analysis -----------------------------------------------------
band(ax, 0.015, 0.010, 0.97, 0.200, "Stage C   Analysis")

for i, (x, t) in enumerate([
        (0.045, "Composition of\ntime to first\nprediction"),
        (0.245, "Equivalence tests\namong pickle,\njoblib and native"),
        (0.445, "Regression of\nsize on load time\nand on total cost"),
        (0.645, "Bootstrap\ncrossover batch\nsize per library"),
        (0.845, "Prediction\nagreement\nbetween runtimes")]):
    box(ax, x, 0.028, 0.135, 0.100, t, fs=FONT - 1.5)

arrow(ax, (0.900, 0.290), (0.900, 0.132), ls="--")

fig.savefig(f"{OUT}/fig_method_pipeline.png", dpi=220,
            bbox_inches="tight", facecolor="white")
plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 2 - cold-start measurement protocol
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.0, 5.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# Parent side
box(ax, 0.020, 0.560, 0.185, 0.300,
    "Orchestrator\nprocess\n\nstarts the clock,\nlaunches the probe,\nstops the clock",
    fs=FONT - 0.5)
ax.text(0.1125, 0.505, "measures total process wall time",
        ha="center", va="center", fontsize=FONT - 1.5, color="#5a6b7c",
        style="italic")

# Isolated child process
ax.add_patch(FancyBboxPatch((0.255, 0.130), 0.725, 0.760,
                            boxstyle="round,pad=0.010",
                            facecolor="#fbf1ec", edgecolor=C_WARM,
                            linewidth=1.8))
ax.text(0.6175, 0.855, "Isolated child process, discarded after one measurement",
        ha="center", va="center", fontsize=FONT + 0.5, weight="bold",
        color=C_WARM)

phases = [
    ("1. Interpreter\nstart-up", "43.8 ms", "#dde3e9"),
    ("2. Import\nnumpy", "75.8 ms", "#cfd8e1"),
    ("3. Import the\nmodel library", "41 to 1963 ms", "#e8a487"),
    ("4. Deserialize\nthe artifact", "1.9 to 70 ms", "#a8c4de"),
    ("5. First\nprediction", "0.5 to 4.3 ms", "#a9cbb0"),
]
x0, w, gap = 0.288, 0.119, 0.014
for i, (name, val, col) in enumerate(phases):
    x = x0 + i * (w + gap)
    box(ax, x, 0.470, w, 0.230, name, fc=col, ec="#7d8b99", fs=FONT - 0.5)
    ax.text(x + w / 2, 0.428, val, ha="center", va="center",
            fontsize=FONT - 1, color="#33475b", weight="bold")
    if i < len(phases) - 1:
        arrow(ax, (x + w, 0.585), (x + w + gap, 0.585), lw=1.1)

arrow(ax, (0.205, 0.585), (0.290, 0.585), lw=1.5)

# Bracket for TTFP
ax.plot([0.288, 0.288, 0.947, 0.947], [0.378, 0.352, 0.352, 0.378],
        color=C_ACCENT, lw=1.3)
ax.text(0.617, 0.318, "Time to first prediction",
        ha="center", va="center", fontsize=FONT + 0.5, weight="bold",
        color=C_ACCENT)

# Steady state
box(ax, 0.288, 0.150, 0.390, 0.105,
    "Then 100 further predictions\non the same batch", fs=FONT - 0.5)
box(ax, 0.718, 0.150, 0.245, 0.105,
    "Steady-state latency\nreported as the median", fs=FONT - 0.5,
    fc="#eef4ec", ec="#4a7c59")
arrow(ax, (0.678, 0.2025), (0.718, 0.2025))

# Controls note
ax.text(0.020, 0.075,
        "Controls applied to every process: thread count pinned to one before "
        "numpy loads; identical input batch;\nfixed random seed; five repeats "
        "per configuration; the median across repeats is reported.",
        ha="left", va="center", fontsize=FONT - 1, color="#5a6b7c")

fig.savefig(f"{OUT}/fig_method_protocol.png", dpi=220,
            bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"wrote {OUT}/fig_method_pipeline.png")
print(f"wrote {OUT}/fig_method_protocol.png")
