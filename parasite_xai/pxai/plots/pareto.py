"""The money figure — faithfulness x cost Pareto frontier (and risk-coverage).

Plots each method as a point in (passes-per-explanation, faithfulness) space,
log-x. Inherently-interpretable + amortized methods cluster at 1 pass with high
faithfulness; KernelSHAP/LIME sit at 1000 passes — the visual core of the thesis.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_faithfulness_vs_cost(rows, out_path: str):
    """rows: list of dicts {method, passes, faithfulness, kind} (kind in {posthoc, ante, amortized})."""
    color = {"posthoc": "#c0392b", "ante": "#27ae60", "amortized": "#2980b9"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        ax.scatter(r["passes"], r["faithfulness"], s=90,
                   color=color.get(r["kind"], "#555"), zorder=3,
                   edgecolor="white", linewidth=1.2)
        ax.annotate(r["method"], (r["passes"], r["faithfulness"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Inference cost — model passes per explanation (log)")
    ax.set_ylabel("Faithfulness (normalised aggregate, higher = better)")
    ax.set_title("Faithfulness vs cost — lightweight interpretable DSS")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=c, label=k)
               for k, c in color.items()]
    ax.legend(handles=handles, title="method type", loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_risk_coverage(curves: dict, out_path: str):
    """curves: {label: (coverage_array, risk_array)}."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (cov, risk) in curves.items():
        ax.plot(cov, risk, label=label, linewidth=2)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk (1 - accuracy)")
    ax.set_title("Risk–coverage — conformal abstention trust layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
