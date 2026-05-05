"""
DataFlix — Evaluation & Visualization Script
Evaluate all saved models, generate Table 1, and produce plots.
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, PROCESSED_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves():
    """Plot training loss and validation RMSE curves."""
    path_a_hist = RESULTS_DIR / "dataflix_path_a_history.json"
    path_b_hist = RESULTS_DIR / "dataflix_path_b_history.json"

    has_a = path_a_hist.exists()
    has_b = path_b_hist.exists()

    if not has_a and not has_b:
        print("No training history found — skipping training curves.")
        return

    n_plots = int(has_a) + int(has_b)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # ── Path A ────────────────────────────────────────────────────
    if has_a:
        with open(path_a_hist) as f:
            hist = json.load(f)

        epochs = range(1, len(hist["train_loss"]) + 1)
        ax = axes[plot_idx]

        ax.plot(epochs, hist["train_loss"], "b-",  linewidth=1.5, label="Train loss (MSE)")
        if hist.get("val_loss"):
            ax.plot(epochs, hist["val_loss"], "r--", linewidth=1.5, label="Val loss (MSE)")

        ax.set_title("Path A (MSE) — loss curves", fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.grid(True, alpha=0.3)

        # Val RMSE on right axis
        if hist.get("val_rmse"):
            ax2 = ax.twinx()
            ax2.plot(epochs, hist["val_rmse"], "g-.", linewidth=1.5,
                     label="Val RMSE", alpha=0.8)
            ax2.set_ylabel("RMSE", color="green")
            ax2.tick_params(axis="y", labelcolor="green")
            ax2.legend(loc="upper right")

        ax.legend(loc="upper left")

        # Annotate best val RMSE
        if hist.get("val_rmse"):
            best_epoch = int(np.argmin(hist["val_rmse"])) + 1
            best_rmse  = min(hist["val_rmse"])
            ax.axvline(best_epoch, color="gray", linestyle=":", alpha=0.5)
            ax.text(best_epoch + 0.2, ax.get_ylim()[1] * 0.98,
                    f"best ep={best_epoch}\nRMSE={best_rmse:.4f}",
                    fontsize=8, va="top", color="gray")

        plot_idx += 1

    # ── Path B ────────────────────────────────────────────────────
    if has_b:
        with open(path_b_hist) as f:
            hist = json.load(f)

        epochs = range(1, len(hist["train_loss"]) + 1)
        ax = axes[plot_idx]
        ax.plot(epochs, hist["train_loss"], "b-", linewidth=1.5, label="BPR loss")
        ax.set_title("Path B (BPR) — loss curve", fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BPR loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# UMAP embeddings
# ─────────────────────────────────────────────────────────────────────────────

def plot_umap_embeddings():
    """Generate UMAP visualization of learned item embeddings."""
    try:
        import umap
    except ImportError:
        print("umap-learn not installed — skipping UMAP plot.")
        print("  Install with: pip install umap-learn")
        return

    model_path = RESULTS_DIR / "dataflix_path_a.pt"
    if not model_path.exists():
        print("No saved Path A model found — skipping UMAP plot.")
        return

    stats_path = PROCESSED_DIR / "stats.json"
    if not stats_path.exists():
        print("stats.json not found — skipping UMAP plot.")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    from src.models.hybrid import DataFlixModel
    model = DataFlixModel(stats["n_users"], stats["n_movies"], path="A")
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=False)
    )
    model.eval()

    item_emb = model.item_embedding.weight.detach().numpy()   # (n_movies, k)

    # Sample up to 5000 items
    n_sample = min(5000, len(item_emb))
    indices  = np.random.choice(len(item_emb), n_sample, replace=False)
    sample   = item_emb[indices]

    print(f"Computing UMAP on {n_sample} item embeddings...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                        metric="cosine", random_state=42)
    coords  = reducer.fit_transform(sample)

    # Primary genre colour — handle both tensor and dict formats
    genre_path = PROCESSED_DIR / "genre_table.pt"
    colors = np.zeros(n_sample, dtype=int)
    if genre_path.exists():
        genre_raw = torch.load(genre_path, weights_only=False)
        if torch.is_tensor(genre_raw):
            # (n_movies, max_genres) — column 0 is primary genre
            for j, idx in enumerate(indices):
                if idx < len(genre_raw):
                    colors[j] = int(genre_raw[idx, 0])
        elif isinstance(genre_raw, dict):
            gmap = genre_raw.get("movie_genre_ids", {})
            for j, idx in enumerate(indices):
                gids = gmap.get(int(idx), [0])
                colors[j] = gids[0] if gids else 0

    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=colors, cmap="tab20", s=4, alpha=0.6)
    ax.set_title("UMAP of item embeddings (coloured by primary genre)", fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.colorbar(sc, ax=ax, label="Genre ID")

    plt.tight_layout()
    out = RESULTS_DIR / "umap_embeddings.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved UMAP plot -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — ablation results
# ─────────────────────────────────────────────────────────────────────────────

def generate_table_1():
    """Generate Table 1 from ablation results — tries both .csv and .json."""
    df = None

    # Try CSV first
    csv_path = RESULTS_DIR / "ablation_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0)

    # Fall back to JSON
    if df is None:
        json_path = RESULTS_DIR / "ablation_results.json"
        if json_path.exists():
            with open(json_path) as f:
                raw = json.load(f)
            df = pd.DataFrame(raw).T   # models as rows, metrics as columns

    if df is None:
        print("No ablation results found — skipping Table 1.")
        print("  (Run training without --skip-ablation to generate these)")
        return

    # Also fold in baseline results if available
    baseline_path = RESULTS_DIR / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baselines = json.load(f)
        # baselines is {name: rmse_float} — expand to match df columns
        for name, val in baselines.items():
            if isinstance(val, (int, float)):
                row = {c: np.nan for c in df.columns}
                if "RMSE" in df.columns:
                    row["RMSE"] = val
                df.loc[name] = row
            elif isinstance(val, dict):
                df.loc[name] = val

    # Select columns to display
    priority = ["RMSE", "MAE", "NDCG@5", "NDCG@10", "NDCG@20",
                "Precision@10", "Recall@10", "MRR", "Coverage", "ILD"]
    display_cols = [c for c in priority if c in df.columns]
    if not display_cols:
        display_cols = df.columns.tolist()

    table = df[display_cols].sort_values(
        display_cols[0] if display_cols else df.columns[0]
    )

    print("\n" + "=" * 80)
    print("TABLE 1: Evaluation Results")
    print("=" * 80)
    print(table.to_string(float_format="%.4f"))

    # Save as LaTeX
    latex_path = RESULTS_DIR / "table_1.tex"
    latex = table.to_latex(float_format="%.4f", bold_rows=True, na_rep="—")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved -> {latex_path}")

    # Save as CSV for easy viewing
    csv_out = RESULTS_DIR / "table_1.csv"
    table.to_csv(csv_out, float_format="%.4f")
    print(f"CSV table saved    -> {csv_out}")

    return table


# ─────────────────────────────────────────────────────────────────────────────
# Ablation bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation_comparison():
    """Horizontal bar chart comparing all model variants by RMSE."""

    # Collect RMSE from both ablation and baseline results
    name_rmse = {}

    for path in [RESULTS_DIR / "ablation_results.json",
                 RESULTS_DIR / "ablation_results.csv"]:
        if not path.exists():
            continue
        if path.suffix == ".json":
            with open(path) as f:
                raw = json.load(f)
            for name, val in raw.items():
                if isinstance(val, dict) and "RMSE" in val:
                    name_rmse[name] = val["RMSE"]
                elif isinstance(val, (int, float)):
                    name_rmse[name] = val
        else:
            df = pd.read_csv(path, index_col=0)
            if "RMSE" in df.columns:
                for name, row in df.iterrows():
                    name_rmse[name] = float(row["RMSE"])

    # Add baselines
    baseline_path = RESULTS_DIR / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baselines = json.load(f)
        for name, val in baselines.items():
            rmse = val["RMSE"] if isinstance(val, dict) else val
            if isinstance(rmse, (int, float)):
                name_rmse[name] = rmse

    if not name_rmse:
        print("No RMSE results found — skipping comparison plot.")
        return

    models = list(name_rmse.keys())
    rmses  = [name_rmse[m] for m in models]

    # Sort by RMSE ascending
    order  = np.argsort(rmses)
    models = [models[i] for i in order]
    rmses  = [rmses[i]  for i in order]

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.5 + 1)))
    palette = sns.color_palette("Blues_r", len(models))
    bars    = ax.barh(models, rmses, color=palette, edgecolor="white", linewidth=0.5)

    # Highlight best (lowest RMSE)
    bars[0].set_color("#2ecc71")
    bars[0].set_edgecolor("#27ae60")
    bars[0].set_linewidth(1.5)

    # Value labels
    for bar, val in zip(bars, rmses):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("RMSE (lower is better)", fontsize=11)
    ax.set_title("Model comparison — RMSE on test set", fontsize=13)
    ax.set_xlim(0, max(rmses) * 1.12)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    out = RESULTS_DIR / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved model comparison -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Score distribution plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_distribution():
    """
    Plot the distribution of predicted scores vs actual ratings on test set.
    Helps diagnose if the model is well-calibrated.
    """
    test_path = PROCESSED_DIR / "test.csv"
    model_path = RESULTS_DIR / "dataflix_path_a.pt"
    if not test_path.exists() or not model_path.exists():
        print("Test CSV or Path A model not found — skipping score distribution.")
        return

    test = pd.read_csv(test_path)
    if len(test) > 50_000:
        test = test.sample(50_000, random_state=42)

    with open(PROCESSED_DIR / "stats.json") as f:
        stats = json.load(f)

    from src.models.hybrid import DataFlixModel
    from src.training.trainer import _build_genre_tensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DataFlixModel(stats["n_users"], stats["n_movies"], path="A").to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False)
    )
    model.eval()

    # Load features
    sbert_data = torch.load(PROCESSED_DIR / "sbert_embeddings.pt", weights_only=False)
    sbert      = (sbert_data["embeddings"] if isinstance(sbert_data, dict)
                  else sbert_data).to(device)
    history    = torch.load(PROCESSED_DIR / "history_embeddings.pt",
                            weights_only=False).to(device)
    pop        = torch.load(PROCESSED_DIR / "popularity.pt",
                            weights_only=False).float().to(device)
    user_feat  = torch.load(PROCESSED_DIR / "user_features.pt",
                            weights_only=False).to(device)
    genre_raw  = torch.load(PROCESSED_DIR / "genre_table.pt", weights_only=False)
    genre_t    = (genre_raw if torch.is_tensor(genre_raw)
                  else _build_genre_tensor(genre_raw, stats["n_movies"])).to(device)

    if pop.dim() == 1:
        pop = pop.unsqueeze(1)

    # Batch inference
    BATCH = 4096
    all_preds = []
    uids = torch.tensor(test["user_idx"].values, dtype=torch.long)
    iids = torch.tensor(test["movie_idx"].values, dtype=torch.long)

    with torch.no_grad():
        for i in range(0, len(uids), BATCH):
            u = uids[i:i+BATCH].to(device)
            it = iids[i:i+BATCH].to(device)
            s  = sbert[it]
            p  = pop[it]
            g  = genre_t[it]
            h  = history[u]
            uf = user_feat[u]
            preds = model(u, it, s, p, g, h, uf)
            all_preds.append(preds.cpu())

    preds_np   = torch.cat(all_preds).numpy()
    rating_col = "rating_centered" if "rating_centered" in test.columns else "rating"
    actuals_np = test[rating_col].values

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram overlay
    axes[0].hist(actuals_np, bins=30, alpha=0.6, label="Actual ratings",
                 color="steelblue", density=True)
    axes[0].hist(preds_np,   bins=30, alpha=0.6, label="Predicted scores",
                 color="coral",     density=True)
    axes[0].set_xlabel("Rating / score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Predicted vs actual rating distribution", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Scatter sample
    n_scatter = min(5000, len(preds_np))
    idx_s = np.random.choice(len(preds_np), n_scatter, replace=False)
    axes[1].scatter(actuals_np[idx_s], preds_np[idx_s],
                    alpha=0.15, s=5, color="steelblue")
    lo = min(actuals_np.min(), preds_np.min())
    hi = max(actuals_np.max(), preds_np.max())
    axes[1].plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect fit")
    axes[1].set_xlabel("Actual rating")
    axes[1].set_ylabel("Predicted score")
    axes[1].set_title("Actual vs predicted (sample)", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    rmse = float(np.sqrt(((preds_np - actuals_np) ** 2).mean()))
    mae  = float(np.abs(preds_np - actuals_np).mean())
    fig.suptitle(f"Path A — Test RMSE: {rmse:.4f}  MAE: {mae:.4f}", fontsize=13)

    plt.tight_layout()
    out = RESULTS_DIR / "score_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved score distribution -> {out}  (RMSE={rmse:.4f}, MAE={mae:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DataFlix — Evaluation & Visualization")
    print("=" * 60)

    print("\n[1/4] Training curves...")
    plot_training_curves()

    print("\n[2/4] Score distribution on test set...")
    plot_score_distribution()

    print("\n[3/4] Table 1 (ablation results)...")
    generate_table_1()

    print("\n[4/4] Model comparison bar chart...")
    plot_ablation_comparison()

    print("\n[Optional] UMAP embeddings (slow — skip if umap-learn not installed)...")
    plot_umap_embeddings()

    print("\n" + "=" * 60)
    print("ALL EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Outputs in: {RESULTS_DIR}")
    print("  training_curves.png")
    print("  score_distribution.png")
    print("  model_comparison.png   (if ablation ran)")
    print("  umap_embeddings.png    (if umap-learn installed)")
    print("  table_1.tex / table_1.csv")


if __name__ == "__main__":
    main()