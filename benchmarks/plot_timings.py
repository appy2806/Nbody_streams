"""Time per particle per force evaluation against N (README figure).

    python benchmarks/plot_timings.py [--data benchmarks/data] [--out plots/acceleration_timings.png]

Reads ``sweep_<label>_<method>.json`` written by ``timing_sweep.py``: CPU
direct and FMM from ``--cpu-label`` (default rtx3080, the laptop), GPU direct
(solid) and GPU tree (dashed) for every GPU label present.  Uses the minimum
call time of each point.  The shaded bands mark the fastest solver on the
consumer GPU (``--cpu-label``), from the measured crossovers.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, MUTED, SURFACE, BAND = "#1f1f1e", "#52524e", "#8a8a85", "#fcfcfb", "#ecebe5"
C_CPU = "#2a78d6"
GPUS = [("rtx3080", "RTX 3080 Laptop", "#eb6834"), ("l40", "L40", "#8a5cc7"),
        ("h200", "H200", "#1baf7a")]


def load(d, label, method):
    f = os.path.join(d, f"sweep_{label}_{method}.json")
    if not os.path.exists(f):
        return None
    pts = json.load(open(f))["points"]
    if not pts:
        return None
    n = np.array([p["n"] for p in pts], float)
    t = np.array([p.get("t_min", p.get("t_call")) for p in pts], float)
    return n, t / n


def crossover(a, b):
    """First N (log-interpolated) where curve b becomes faster than curve a."""
    if a is None or b is None:
        return None
    n = np.intersect1d(a[0], b[0])
    if len(n) < 2:
        return None
    ra = np.interp(np.log(n), np.log(a[0]), np.log(a[1]))
    rb = np.interp(np.log(n), np.log(b[0]), np.log(b[1]))
    d = ra - rb                                   # > 0: b faster
    k = np.nonzero((d[:-1] <= 0) & (d[1:] > 0))[0]
    if not len(k):
        return None
    i = k[-1]
    f = -d[i] / (d[i + 1] - d[i])
    return float(np.exp(np.log(n[i]) + f * (np.log(n[i + 1]) - np.log(n[i]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "plots", "acceleration_timings.png"))
    ap.add_argument("--cpu-label", default="rtx3080")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.linewidth": 0.8, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
    kw = dict(lw=1.9, solid_capstyle="round", dash_capstyle="round")

    cpu = load(args.data, args.cpu_label, "cpu_direct")
    fmm = load(args.data, args.cpu_label, "fmm")
    if cpu is not None:
        ax.plot(*cpu, "-", color=C_CPU, label="direct: CPU, 16 threads", **kw)
    if fmm is not None:
        ax.plot(*fmm, "--", color=C_CPU, label="FMM (falcON): CPU, 1 core", **kw)
    ends = []
    for lab, name, c in GPUS:
        d = load(args.data, lab, "gpu_direct")
        t = load(args.data, lab, "gpu_tree")
        if d is not None:
            ax.plot(*d, "-", color=c, label=f"direct: {name}", **kw)
        if t is not None:
            ax.plot(*t, "--", color=c, label=f"tree: {name}", **kw)
            ends.append((t[0][-1], t[1][-1], name.split()[0] + " tree", {"l40": 5, "h200": -4}.get(lab, 0)))

    # Fastest-solver bands on the consumer GPU.
    gd = load(args.data, args.cpu_label, "gpu_direct")
    gt = load(args.data, args.cpu_label, "gpu_tree")
    x1, x2 = crossover(cpu, gd), crossover(gd, gt)
    lo, hi = 1e2, 1e9
    edges = [lo] + [x for x in (x1, x2) if x] + [hi]
    names = ["CPU\ndirect", "GPU\ndirect", "GPU tree $\\rightarrow$"][-(len(edges) - 1):]
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        if i % 2 == 0:
            ax.axvspan(a, b, color=BAND, zorder=0, lw=0)
        if i:
            ax.axvline(a, color="#cfcec8", lw=0.8, zorder=0.5)
        xm = np.sqrt(a * min(b, 1e5 if i == len(edges) - 2 else b))
        ax.text(xm, 2.4e-9, names[i], ha="center", va="bottom", fontsize=9.5, color=INK2)

    for n, r, txt, dy in ends:
        ax.annotate(txt, (n, r), xytext=(5, dy), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK2)
    ax.axhline(1e-6, color="#d2d1cb", lw=0.8, ls=":", zorder=0.5)
    ax.text(1.2e8, 1.1e-6, "1 $\\mu$s per particle", fontsize=8.5, color=MUTED, va="bottom", ha="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(2e-9, 1e-3)
    ax.grid(True, which="major", color="#e2e1dc", lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("Number of particles $N$")
    ax.set_ylabel("Time per particle per force evaluation (s)")
    ax.set_title("Force evaluation cost vs $N$", color=INK, fontsize=12)
    leg = ax.legend(ncol=2, fontsize=7.8, loc="upper center", frameon=True, columnspacing=1.2,
                    handlelength=2.6, borderpad=0.5)
    leg.get_frame().set_edgecolor("#d8d7d1"); leg.get_frame().set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}; crossovers on {args.cpu_label}: CPU->GPU direct "
          f"{x1 and round(x1)}, GPU direct->tree {x2 and round(x2)}")


if __name__ == "__main__":
    main()
