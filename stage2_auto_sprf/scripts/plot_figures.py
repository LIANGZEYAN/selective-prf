#!/usr/bin/env python3
"""
Generate two publication-quality figures for the paper:
  1. Heatmap: AUC across models × aggregations, grouped by paradigm
  2. Scatter plot: AUC vs inference cost for all methods
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Load h=0 results ────────────────────────────────────────────────────────
df = pd.read_csv(PROJECT_ROOT / "results" / "summary_table_h000.csv")

# =============================================================================
# FIGURE 1: HEATMAP — AUC across models × aggregation × paradigm
# =============================================================================

def make_heatmap():
    """Build a multi-panel heatmap: single-run (left) and SC (right)."""

    # ── Define the model/method rows we want, in display order ──────────
    # Single-run methods
    single_run_models = [
        # (display_name, method_prefix_in_csv)
        # Pairwise
        ("Pw-Qwen2.5-7B",   "Pairwise-Qwen-7B"),
        ("Pw-Qwen2.5-14B",  "Pairwise-Qwen-14B"),
        ("Pw-Qwen2.5-72B",  "Pairwise-Qwen-72B"),
        ("Pw-Qwen3-8B",     "Pairwise-Qwen3-8B"),
        ("Pw-Qwen3-14B",    "Pairwise-Qwen3-14B"),
        ("Pw-LLaMA-8B",     "Pairwise-Llama-8B"),
        ("Pw-LLaMA-70B",    "Pairwise-Llama-70B"),
        ("Pw-Mistral-7B",   "Pairwise-Mistral-7B"),
        ("Pw-Mistral-24B",  "Pairwise-Mistral-24B"),
        ("Pw-RankZephyr-7B","Pairwise-RankZephyr-7B"),
        # Setwise
        ("Sw-Qwen2.5-7B",   "Setwise-Qwen-7B"),
        ("Sw-Qwen2.5-14B",  "Setwise-Qwen-14B"),
        ("Sw-Qwen2.5-72B",  "Setwise-Qwen-72B"),
        ("Sw-R1-7B",        "Setwise-R1-7B"),
        ("Sw-R1-14B",       "Setwise-R1-14B"),
        ("Sw-SFT-7B",       "Setwise-SFT-7B"),
        ("Sw-SFT-14B",      "Setwise-SFT-14B"),
        ("Sw-Qwen3-8B",     "Setwise-Qwen3-8B"),
        ("Sw-Qwen3-14B",    "Setwise-Qwen3-14B"),
        ("Sw-Qwen3-8B-Thk", "Setwise-Qwen3-8B-Think"),
        ("Sw-Qwen3-14B-Thk","Setwise-Qwen3-14B-Think"),
        ("Sw-LLaMA-8B",     "Setwise-Llama-8B"),
        ("Sw-LLaMA-70B",    "Setwise-Llama-70B"),
        ("Sw-Mistral-7B",   "Setwise-Mistral-7B"),
        ("Sw-Mistral-24B",  "Setwise-Mistral-24B"),
        ("Sw-RankZephyr-7B","Setwise-RankZephyr-7B"),
        # Pointwise
        ("Pt-Reranker-4B",  "Pointwise-Qwen3-Reranker-4B"),
        ("Pt-Reranker-8B",  "Pointwise-Qwen3-Reranker-8B"),
    ]

    # SC methods
    sc_models = [
        ("SC-Pw-Qwen2.5-14B",  "SC-Pairwise-Qwen-14B"),
        ("SC-Pw-Qwen2.5-72B",  "SC-Pairwise-Qwen-72B"),
        ("SC-Pw-Qwen3-8B",     "SC-Pairwise-Qwen3-8B"),
        ("SC-Pw-LLaMA-70B",    "SC-Pairwise-Llama-70B"),
        ("SC-Sw-Qwen2.5-14B",  "SC-Setwise-Qwen-14B"),
        ("SC-Sw-R1-7B",        "SC-Setwise-R1-7B"),
        ("SC-Sw-R1-14B",       "SC-Setwise-R1-14B"),
        ("SC-Sw-SFT-7B",       "SC-Setwise-SFT-7B"),
        ("SC-Sw-SFT-14B",      "SC-Setwise-SFT-14B"),
        ("SC-Sw-Qwen3-8B",     "SC-Setwise-Qwen3-8B"),
        ("SC-Sw-Qwen3-14B",    "SC-Setwise-Qwen3-14B"),
        ("SC-Sw-Qwen3-8B-Thk", "SC-Setwise-Qwen3-8B-Think"),
        ("SC-Sw-Qwen3-14B-Thk","SC-Setwise-Qwen3-14B-Think"),
        ("SC-Sw-LLaMA-70B",    "SC-Setwise-Llama-70B"),
        ("SC-Sw-Mistral-7B",   "SC-Setwise-Mistral-7B"),
        ("SC-Sw-Mistral-24B",  "SC-Setwise-Mistral-24B"),
    ]

    agg_cols = ["DCG", "MRR", "Majority@k"]

    def build_matrix(model_list):
        """Extract AUC values into a (n_models, 3) matrix."""
        matrix = np.full((len(model_list), 3), np.nan)
        for i, (display, prefix) in enumerate(model_list):
            for j, agg in enumerate(agg_cols):
                name = f"{prefix}-{agg}"
                row = df[df["method_name"] == name]
                if not row.empty:
                    val = row["auc"].values[0]
                    if pd.notna(val):
                        matrix[i, j] = val
        return matrix

    mat_single = build_matrix(single_run_models)
    mat_sc = build_matrix(sc_models)

    # ── Plot ────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 12),
                                    gridspec_kw={"width_ratios": [28, 16], "wspace": 0.55})

    vmin, vmax = 0.40, 0.85
    cmap = plt.cm.RdYlGn

    # Helper to draw one heatmap
    def draw_heatmap(ax, matrix, labels, title, show_cbar=False):
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xticks(range(3))
        ax.set_xticklabels(agg_cols, fontsize=9, fontweight="bold")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([l[0] for l in labels], fontsize=7.5)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        # Annotate cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=6.5, color="gray")
                else:
                    color = "white" if val < 0.50 or val > 0.78 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=6.5, fontweight="bold" if val >= 0.78 else "normal",
                            color=color)

        # Draw horizontal separators for paradigm groups
        return im

    im1 = draw_heatmap(ax1, mat_single, single_run_models, "Single-Run Methods")
    # Add separator lines between pairwise/setwise/pointwise
    for y in [9.5, 25.5]:  # after pairwise, after setwise
        ax1.axhline(y=y, color="black", linewidth=1.5)
    # Add paradigm labels — placed further left to avoid overlapping model names
    ax1.text(-1.2, 4.5, "Pairwise", rotation=90, va="center", ha="center",
             fontsize=8, fontweight="bold", color="#444")
    ax1.text(-1.2, 17.5, "Setwise", rotation=90, va="center", ha="center",
             fontsize=8, fontweight="bold", color="#444")
    ax1.text(-1.2, 26.5, "Pointwise", rotation=90, va="center", ha="center",
             fontsize=8, fontweight="bold", color="#444")

    im2 = draw_heatmap(ax2, mat_sc, sc_models, "Self-Consistency (R=5)")
    # Separator between SC-Pairwise and SC-Setwise
    ax2.axhline(y=3.5, color="black", linewidth=1.5)
    ax2.text(-1.1, 1.5, "Pairwise", rotation=90, va="center", ha="center",
             fontsize=8, fontweight="bold", color="#444")
    ax2.text(-1.1, 9.5, "Setwise", rotation=90, va="center", ha="center",
             fontsize=8, fontweight="bold", color="#444")

    # Colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.3, pad=0.02, aspect=20)
    cbar.set_label("AUC", fontsize=10, fontweight="bold")

    fig.suptitle("AUC by Model, Aggregation, and Paradigm ($h{=}0$, $N{=}38$)",
                 fontsize=13, fontweight="bold", y=0.98)

    plt.savefig(OUT_DIR / "heatmap_auc.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUT_DIR / "heatmap_auc.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved heatmap → {OUT_DIR / 'heatmap_auc.pdf'}")


# =============================================================================
# FIGURE 2: SCATTER — AUC vs Inference Cost
# =============================================================================

def make_scatter():
    """Scatter plot of AUC vs approximate inference cost."""

    # ── Define methods with approximate cost estimates ───────────────────
    # Cost = relative GPU-seconds per query (approximate)
    # For LLM methods: cost ∝ model_params × n_calls × n_runs
    # n=9 docs, heapsort ≈ O(n log n) ≈ 28 comparisons for pairwise,
    # setwise ≈ 14 calls, pointwise = 9 calls
    # SC multiplies by 5

    methods = []

    # Helper: pick best agg AUC for a method prefix
    def best_auc(prefix):
        rows = df[df["method_name"].str.startswith(prefix + "-")]
        if rows.empty:
            return None
        return rows["auc"].max()

    # ── Baselines (zero or near-zero cost) ──────────────────────────────
    baselines = [
        ("TREC nDCG@10", 0.01, 0.681, "baselines"),
        ("NED", 0.02, 0.556, "baselines"),
        ("IDF Coverage", 0.02, 0.539, "baselines"),
        ("QTC", 0.05, 0.439, "baselines"),
        ("WIG", 0.01, 0.569, "baselines"),
        ("NQC", 0.01, 0.414, "baselines"),
        ("Clarity", 0.01, 0.475, "baselines"),
        ("SMV", 0.01, 0.583, "baselines"),
        ("TDC", 0.05, 0.544, "baselines"),
    ]
    methods.extend(baselines)

    # ── Direct classifiers (1a) — query-only, cheap ─────────────────────
    # All AUC values at h=0, N=38
    classifiers_1a = [
        ("n_terms", 0.001, 0.714, "query-features"),
        ("TermSimMean\n(MiniLM)", 0.03, 0.697, "query-features"),
        ("TermSimMean\n(GTE)", 0.08, 0.719, "query-features"),
        ("TermSimStd\n(GTE)", 0.08, 0.736, "query-features"),
        ("TermSimMean\n(BGE)", 0.08, 0.775, "query-features"),
    ]
    methods.extend(classifiers_1a)

    # ── Cross-encoder features (2b) — moderate cost ─────────────────────
    # All AUC values at h=0, N=38
    classifiers_2b = [
        ("CE score slope\n(ms-marco)", 0.15, 0.681, "cross-encoder"),
        ("Reranker entropy\n(Qwen3-8B)", 2.0, 0.678, "cross-encoder"),
    ]
    methods.extend(classifiers_2b)

    # ── Single-run LLM methods — pick best agg per model ────────────────
    # Approximate GPU-seconds per query for single run
    # 7B pairwise: ~8s, 14B pairwise: ~15s, 72B pairwise: ~45s (API)
    # 7B setwise: ~5s, 14B setwise: ~10s, 72B setwise: ~30s
    # Pointwise 4B: ~2s, 8B: ~4s

    single_run_specs = [
        # (label, prefix, cost, category)
        ("Pw-Qwen2.5-7B", "Pairwise-Qwen-7B", 8, "pairwise"),
        ("Pw-Qwen2.5-14B", "Pairwise-Qwen-14B", 15, "pairwise"),
        ("Pw-Qwen2.5-72B", "Pairwise-Qwen-72B", 45, "pairwise"),
        ("Pw-Qwen3-8B", "Pairwise-Qwen3-8B", 8, "pairwise"),
        ("Pw-Qwen3-14B", "Pairwise-Qwen3-14B", 15, "pairwise"),
        ("Pw-LLaMA-70B", "Pairwise-Llama-70B", 45, "pairwise"),
        ("Pw-Mistral-24B", "Pairwise-Mistral-24B", 20, "pairwise"),
        ("Sw-Qwen2.5-7B", "Setwise-Qwen-7B", 5, "setwise"),
        ("Sw-Qwen2.5-14B", "Setwise-Qwen-14B", 10, "setwise"),
        ("Sw-Qwen2.5-72B", "Setwise-Qwen-72B", 30, "setwise"),
        ("Sw-R1-7B", "Setwise-R1-7B", 6, "setwise"),
        ("Sw-R1-14B", "Setwise-R1-14B", 12, "setwise"),
        ("Sw-SFT-7B", "Setwise-SFT-7B", 6, "setwise"),
        ("Sw-SFT-14B", "Setwise-SFT-14B", 12, "setwise"),
        ("Sw-Qwen3-8B", "Setwise-Qwen3-8B", 6, "setwise"),
        ("Sw-Qwen3-14B", "Setwise-Qwen3-14B", 12, "setwise"),
        ("Sw-Qwen3-14B-Thk", "Setwise-Qwen3-14B-Think", 20, "setwise"),
        ("Sw-LLaMA-70B", "Setwise-Llama-70B", 30, "setwise"),
        ("Sw-Mistral-7B", "Setwise-Mistral-7B", 5, "setwise"),
        ("Sw-Mistral-24B", "Setwise-Mistral-24B", 15, "setwise"),
        ("Pt-Reranker-4B", "Pointwise-Qwen3-Reranker-4B", 2, "pointwise"),
        ("Pt-Reranker-8B", "Pointwise-Qwen3-Reranker-8B", 4, "pointwise"),
    ]

    for label, prefix, cost, cat in single_run_specs:
        auc = best_auc(prefix)
        if auc is not None:
            methods.append((label, cost, auc, cat))

    # ── SC methods — 5× cost of single run ──────────────────────────────
    sc_specs = [
        ("SC-Pw-Qwen2.5-14B", "SC-Pairwise-Qwen-14B", 75, "sc-pairwise"),
        ("SC-Pw-Qwen2.5-72B", "SC-Pairwise-Qwen-72B", 225, "sc-pairwise"),
        ("SC-Sw-R1-7B", "SC-Setwise-R1-7B", 30, "sc-setwise"),
        ("SC-Sw-R1-14B", "SC-Setwise-R1-14B", 60, "sc-setwise"),
        ("SC-Sw-SFT-7B", "SC-Setwise-SFT-7B", 30, "sc-setwise"),
        ("SC-Sw-SFT-14B", "SC-Setwise-SFT-14B", 60, "sc-setwise"),
        ("SC-Sw-Qwen3-8B", "SC-Setwise-Qwen3-8B", 30, "sc-setwise"),
        ("SC-Sw-Qwen3-14B", "SC-Setwise-Qwen3-14B", 60, "sc-setwise"),
        ("SC-Sw-Mistral-7B", "SC-Setwise-Mistral-7B", 25, "sc-setwise"),
        ("SC-Sw-Mistral-24B", "SC-Setwise-Mistral-24B", 75, "sc-setwise"),
    ]

    for label, prefix, cost, cat in sc_specs:
        auc = best_auc(prefix)
        if auc is not None:
            methods.append((label, cost, auc, cat))

    # ── Build DataFrame ─────────────────────────────────────────────────
    plot_df = pd.DataFrame(methods, columns=["label", "cost", "auc", "category"])

    # ── Style mapping ───────────────────────────────────────────────────
    cat_styles = {
        "baselines":      {"color": "#999999", "marker": "s",  "s": 40,  "alpha": 0.7, "zorder": 2},
        "query-features": {"color": "#e377c2", "marker": "D",  "s": 60,  "alpha": 0.9, "zorder": 4},
        "cross-encoder":  {"color": "#ff7f0e", "marker": "^",  "s": 60,  "alpha": 0.9, "zorder": 4},
        "pairwise":       {"color": "#1f77b4", "marker": "o",  "s": 50,  "alpha": 0.8, "zorder": 3},
        "setwise":        {"color": "#2ca02c", "marker": "o",  "s": 50,  "alpha": 0.8, "zorder": 3},
        "pointwise":      {"color": "#9467bd", "marker": "P",  "s": 60,  "alpha": 0.9, "zorder": 4},
        "sc-pairwise":    {"color": "#1f77b4", "marker": "*",  "s": 120, "alpha": 0.9, "zorder": 5},
        "sc-setwise":     {"color": "#2ca02c", "marker": "*",  "s": 120, "alpha": 0.9, "zorder": 5},
    }

    # ── Plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Random chance line
    ax.axhline(y=0.5, color="#ccc", linestyle="--", linewidth=1, zorder=1)
    ax.text(0.012, 0.505, "Random chance", fontsize=7, color="#aaa", style="italic")

    # TREC baseline line
    ax.axhline(y=0.681, color="#ddd", linestyle=":", linewidth=1, zorder=1)
    ax.text(0.012, 0.686, "TREC nDCG@10 baseline", fontsize=7, color="#bbb", style="italic")

    for cat, style in cat_styles.items():
        sub = plot_df[plot_df["category"] == cat]
        if sub.empty:
            continue
        ax.scatter(sub["cost"], sub["auc"], **style, edgecolors="white", linewidth=0.5)

    # Label key methods with manual offsets to avoid overlap
    label_offsets = {
        "n_terms":                      (6, -10, "left"),
        "TermSimMean\n(BGE)":           (6, 5, "left"),
        "Sw-Qwen3-14B":                (-8, 6, "right"),
        "SC-Sw-SFT-7B":                (-8, 7, "right"),
        "SC-Sw-R1-14B":                (6, -10, "left"),
        "SC-Sw-Qwen3-8B":              (6, 5, "left"),
        "Pt-Reranker-8B":              (-8, -10, "right"),
        "Pw-Qwen2.5-14B":              (6, -10, "left"),
        "SC-Pw-Qwen2.5-14B":           (-8, -10, "right"),
        "Sw-Mistral-24B":              (-8, -10, "right"),
        "CE score slope\n(ms-marco)":   (6, -3, "left"),
    }
    for _, row in plot_df.iterrows():
        if row["label"] in label_offsets:
            dx, dy, ha = label_offsets[row["label"]]
            ax.annotate(
                row["label"], (row["cost"], row["auc"]),
                textcoords="offset points", xytext=(dx, dy),
                fontsize=6.5, ha=ha, color="#333",
                arrowprops=dict(arrowstyle="-", color="#ccc", lw=0.5) if row["cost"] > 1 else None,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Approximate Inference Cost (GPU-seconds per query, log scale)",
                   fontsize=10, fontweight="bold")
    ax.set_ylabel("AUC", fontsize=10, fontweight="bold")
    ax.set_title("AUC vs. Inference Cost ($h{=}0$, $N{=}38$)",
                  fontsize=12, fontweight="bold")
    ax.set_ylim(0.38, 0.86)
    ax.set_xlim(0.0008, 400)
    ax.grid(True, alpha=0.2)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#999999",
               markersize=7, label="Baselines (QPP, Pre-hoc)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#e377c2",
               markersize=7, label="Query Features (1a)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#ff7f0e",
               markersize=7, label="Cross-Encoder Features (2b)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markersize=7, label="Pairwise LLM"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markersize=7, label="Setwise LLM"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor="#9467bd",
               markersize=8, label="Pointwise LLM"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#1f77b4",
               markersize=10, label="SC-Pairwise"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#2ca02c",
               markersize=10, label="SC-Setwise"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7.5,
              framealpha=0.9, edgecolor="#ccc", ncol=2)

    plt.savefig(OUT_DIR / "scatter_auc_cost.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUT_DIR / "scatter_auc_cost.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved scatter → {OUT_DIR / 'scatter_auc_cost.pdf'}")


if __name__ == "__main__":
    make_heatmap()
    make_scatter()
    print("Done.")
