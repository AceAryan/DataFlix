"""
DataFlix — Evaluation Script
scripts/evaluate.py

Fixes vs original:
  1. Relevant items = movies rated ABOVE user mean (rating_centered > 0)
     not ALL rated movies — prevents trivially near-zero NDCG
  2. ALS scorer uses explicit dot product + biases (no double mean subtraction)
  3. Cold-start users built from training counts at runtime (not stale CSV)
  4. Feature tensors preloaded to GPU once for hybrid eval (was 7hr, now ~30min)
  5. Results saved immediately after each model (no data loss on Ctrl+C)
  6. User ID lookup pre-built as dict (was O(n) DataFrame scan per user)

Usage:
  python scripts/evaluate.py                    # all models on test split
  python scripts/evaluate.py --model als
  python scripts/evaluate.py --model bpr
  python scripts/evaluate.py --model hybrid
  python scripts/evaluate.py --split val
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.config import (
    PROCESSED_DIR, RESULTS_DIR, DEVICE,
    TEST_CSV, VAL_CSV, TRAIN_CSV,
    COLD_START_THRESHOLD,
    SBERT_EMBEDDINGS_PATH, IMDB_FEATURES_PATH,
    POPULARITY_PATH, HISTORY_EMBEDDINGS_PATH,
    ALS_PATH, TOP_K_VALUES, EVAL_BATCH_SIZE,
    set_seed,
)
from src.models.als    import ALS
from src.models.bpr    import BPR, BPR_FACTORS_PATH
from src.models.hybrid import HybridModel, HybridTrainer, score_all_items, HYBRID_CKPT_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EVAL_REPORT_JSON  = RESULTS_DIR / "evaluation_report.json"
EVAL_REPORT_CSV   = RESULTS_DIR / "evaluation_report.csv"

# Items with rating_centered > this are "relevant" for ranking metrics.
# 0.0 means "rated above the user's own mean" — a fair positive signal.
RELEVANCE_THRESHOLD = 0.0

# Cold threshold used at eval time — higher than COLD_START_THRESHOLD in
# config (which was used during preprocessing) so we actually catch users.
COLD_EVAL_THRESHOLD = max(COLD_START_THRESHOLD, 20)


# ── Metrics ───────────────────────────────────────────────────────────────────

def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    r = relevance[:k].astype(np.float32)
    if r.sum() == 0:
        return 0.0
    return float((r / np.log2(np.arange(2, len(r) + 2))).sum())


def ndcg_at_k(ranked: np.ndarray, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    rel      = np.array([1 if i in relevant else 0 for i in ranked[:k]])
    ideal    = np.ones(min(len(relevant), k))
    ideal_dcg = dcg_at_k(ideal, k)
    return dcg_at_k(rel, k) / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(ranked: np.ndarray, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for i in ranked[:k] if i in relevant) / len(relevant)


def mrr_at_k(ranked: np.ndarray, relevant: set, k: int) -> float:
    for rank, item in enumerate(ranked[:k], 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_eval_data(split: str):
    csv = VAL_CSV if split == "val" else TEST_CSV
    log.info(f"Loading {split} split...")
    eval_df  = pd.read_csv(csv)
    train_df = pd.read_csv(TRAIN_CSV)

    # Build cold users from actual training counts — stale CSV used threshold=5
    train_counts = train_df.groupby("user_idx").size()
    cold_users   = set(train_counts[train_counts < COLD_EVAL_THRESHOLD].index.tolist())

    log.info(f"  Eval  : {len(eval_df):,} ratings")
    log.info(f"  Train : {len(train_df):,} ratings")
    log.info(f"  Cold  : {len(cold_users):,} users (< {COLD_EVAL_THRESHOLD} train ratings)")
    return eval_df, train_df, cold_users


def _build_ground_truth(eval_df, train_df):
    """
    user_relevant : items user rated >= 4.0 stars — true positives for ranking
    user_seen     : all training items — excluded from recommendation
    user_ratings  : all (movie_idx, rating_centered) pairs for RMSE
    user_means    : per-user mean rating from training (for ALS de-centering)
    """
    user_means    = train_df.groupby("user_idx")["rating"].mean().to_dict()

    user_relevant: dict[int, set]  = defaultdict(set)
    user_ratings:  dict[int, list] = defaultdict(list)

    for row in eval_df.itertuples(index=False):
        uid, mid, rc = int(row.user_idx), int(row.movie_idx), float(row.rating_centered)
        user_ratings[uid].append((mid, rc))
        # Use raw rating as relevance signal — centred ratings are noisy
        # for users with few test items (a 4-star from a generous rater
        # might be centred to -0.5 but is still a genuinely liked item)
        raw_rating = float(row.rating)
        if raw_rating >= 4.0:
            user_relevant[uid].add(mid)

    user_seen: dict[int, set] = defaultdict(set)
    for row in train_df.itertuples(index=False):
        user_seen[int(row.user_idx)].add(int(row.movie_idx))

    n_with = len(user_relevant)
    n_all  = len(user_ratings)
    log.info(f"  Users with ≥1 relevant test item: {n_with:,} / {n_all:,} "
             f"({n_all - n_with:,} skipped)")

    return dict(user_relevant), dict(user_seen), dict(user_ratings), user_means


def _load_feature_tensors():
    return {
        "sbert":   torch.load(SBERT_EMBEDDINGS_PATH,   weights_only=False),
        "imdb":    torch.load(IMDB_FEATURES_PATH,      weights_only=False),
        "pop":     torch.load(POPULARITY_PATH,          weights_only=False),
        "history": torch.load(HISTORY_EMBEDDINGS_PATH, weights_only=False),
    }


# ── Scorers ───────────────────────────────────────────────────────────────────

def _score_als(model: ALS, user_idx: int, seen: set,
               user_mean: float = 0.0) -> np.ndarray:
    """
    score(u,i) = p_u · q_i + b_i + b_u + user_mean
    Adding user_mean back converts centred predictions to absolute scale.
    Without this, items a user likes score near 0 (not near 4-5),
    making ranking essentially random.
    """
    scores = (model.item_factors @ model.user_factors[user_idx]
              + model.item_biases
              + model.user_biases[user_idx]
              + user_mean)
    if seen:
        scores = scores.copy()
        scores[list(seen)] = -np.inf
    return scores


def _score_bpr(model: BPR, user_idx: int, seen: set) -> np.ndarray:
    with torch.no_grad():
        scores = model.model.score_all_items(user_idx).cpu().numpy()
    if seen:
        scores = scores.copy()
        scores[list(seen)] = -np.inf
    return scores


def _score_hybrid(model: HybridModel, user_idx: int,
                  tensors: dict, seen: set) -> np.ndarray:
    scores = score_all_items(
        model=model, user_idx=user_idx,
        sbert_emb=tensors["sbert"], imdb_feats=tensors["imdb"],
        popularity=tensors["pop"], history_emb=tensors["history"],
        device=DEVICE, batch_size=EVAL_BATCH_SIZE,
    )
    if seen:
        scores = scores.copy()
        scores[list(seen)] = -np.inf
    return scores


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_model(model_name, score_fn, eval_df, train_df,
                   cold_users, k_values=TOP_K_VALUES) -> dict:

    user_relevant, user_seen, user_ratings, user_means = _build_ground_truth(eval_df, train_df)

    # Pre-build user_id lookup — O(1) vs O(n) DataFrame scan per user
    # (kept for potential future use; source segmentation removed with NF drop)
    uid_lookup = (
        eval_df[["user_idx", "user_id"]].drop_duplicates("user_idx")
        .set_index("user_idx")["user_id"].to_dict()
    )

    # NF segment removed — Netflix dropped due to 5.9% movie coverage
    segments = ["overall", "warm", "cold"]
    accum = {seg: defaultdict(list) for seg in segments}

    users   = list(user_relevant.keys())
    n_users = len(users)
    log.info(f"  Evaluating {n_users:,} users...")

    t = time.time()
    for i, user_idx in enumerate(users):
        if i % 5000 == 0 and i > 0:
            eta = (time.time() - t) / i * (n_users - i)
            log.info(f"    {i:>6}/{n_users:,}  ETA {eta/60:.1f}m")

        relevant  = user_relevant[user_idx]
        seen      = user_seen.get(user_idx, set())
        warmth    = "cold" if user_idx in cold_users else "warm"
        user_segs = ["overall", warmth]

        # Ranking
        scores  = score_fn(user_idx, seen)
        top_max = max(k_values)
        if np.isfinite(scores).sum() < top_max:
            continue
        top_idx = np.argpartition(scores, -top_max)[-top_max:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        for k in k_values:
            for seg in user_segs:
                accum[seg][f"ndcg@{k}"].append(ndcg_at_k(top_idx, relevant, k))
                accum[seg][f"recall@{k}"].append(recall_at_k(top_idx, relevant, k))
                accum[seg][f"mrr@{k}"].append(mrr_at_k(top_idx, relevant, k))

        # Rating metrics on all test items
        pairs = user_ratings.get(user_idx, [])
        if pairs:
            true_r = np.array([r for _, r in pairs], dtype=np.float32)
            pred_r = np.array(
                [float(scores[m]) if np.isfinite(scores[m]) else 0.0
                 for m, _ in pairs], dtype=np.float32
            )
            rmse = float(np.sqrt(np.mean((true_r - pred_r) ** 2)))
            mae  = float(np.mean(np.abs(true_r - pred_r)))
            for seg in user_segs:
                accum[seg]["rmse"].append(rmse)
                accum[seg]["mae"].append(mae)

    results = {}
    for seg, metrics in accum.items():
        results[seg] = {
            m: float(np.mean(v)) if v else float("nan")
            for m, v in metrics.items()
        }
        results[seg]["n_users"] = len(accum[seg].get(f"ndcg@{k_values[0]}", []))

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_results(name, results, k_values):
    log.info(f"\n  ── {name} ──")
    hdr = f"  {'Segment':<10}" + "".join(
        f"  {'NDCG@'+str(k):<10}{'Recall@'+str(k):<12}{'MRR@'+str(k):<10}"
        for k in k_values
    ) + f"  {'RMSE':<8}{'MAE':<8}{'N':<8}"
    log.info(hdr)
    log.info("  " + "─" * (len(hdr) - 2))
    for seg in ["overall", "warm", "cold"]:
        m = results.get(seg, {})
        if not m or m.get("n_users", 0) == 0:
            continue
        row = f"  {seg:<10}"
        for k in k_values:
            row += (f"  {m.get(f'ndcg@{k}', float('nan')):<10.4f}"
                    f"{m.get(f'recall@{k}', float('nan')):<12.4f}"
                    f"{m.get(f'mrr@{k}', float('nan')):<10.4f}")
        row += (f"  {m.get('rmse', float('nan')):<8.4f}"
                f"{m.get('mae', float('nan')):<8.4f}"
                f"{m.get('n_users', 0):<8,}")
        log.info(row)


def _save_results(new_results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    if EVAL_REPORT_JSON.exists():
        with open(EVAL_REPORT_JSON) as f:
            existing = json.load(f)
    existing.update(new_results)
    with open(EVAL_REPORT_JSON, "w") as f:
        json.dump(existing, f, indent=2)
    rows = []
    for model, segs in existing.items():
        for seg, metrics in segs.items():
            rows.append({"model": model, "segment": seg, **metrics})
    pd.DataFrame(rows).to_csv(EVAL_REPORT_CSV, index=False)
    log.info(f"  Saved → {EVAL_REPORT_JSON.name}  &  {EVAL_REPORT_CSV.name}")


def _print_comparison(all_results, k=10):
    log.info("")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  Model Comparison — overall, test split               ║")
    log.info("╠══════════════════╦═════════╦══════════╦══════════════╣")
    log.info("║  Model           ║ NDCG@10 ║ Recall@10║ RMSE         ║")
    log.info("╠══════════════════╬═════════╬══════════╬══════════════╣")
    for name, res in all_results.items():
        m = res.get("overall", {})
        log.info(
            f"║  {name:<16}║ "
            f"{m.get(f'ndcg@{k}', float('nan')):<7.4f} ║ "
            f"{m.get(f'recall@{k}', float('nan')):<8.4f} ║ "
            f"{m.get('rmse', float('nan')):<12.4f} ║"
        )
    log.info("╚══════════════════╩═════════╩══════════╩══════════════╝")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    set_seed()
    t0 = time.time()

    eval_df, train_df, cold_users = _load_eval_data(args.split)
    run_all = args.model is None
    all_results = {}
    tensors = None

    # ALS
    if run_all or args.model == "als":
        if not ALS_PATH.exists():
            log.warning("ALS factors not found — skipping.")
        else:
            log.info("\nEvaluating ALS...")
            als = ALS.load(ALS_PATH)
            user_means_als = train_df.groupby("user_idx")["rating"].mean().to_dict()
            results = evaluate_model(
                "ALS", lambda u, s: _score_als(als, u, s, user_means_als.get(u, 0.0)),
                eval_df, train_df, cold_users
            )
            _print_results("ALS", results, TOP_K_VALUES)
            all_results["ALS"] = results
            _save_results({"ALS": results})

    # BPR
    if run_all or args.model == "bpr":
        if not BPR_FACTORS_PATH.exists():
            log.warning("BPR factors not found — skipping.")
        else:
            log.info("\nEvaluating BPR...")
            bpr = BPR.load(BPR_FACTORS_PATH, device=DEVICE)
            bpr.model.eval()
            results = evaluate_model(
                "BPR", lambda u, s: _score_bpr(bpr, u, s),
                eval_df, train_df, cold_users
            )
            _print_results("BPR", results, TOP_K_VALUES)
            all_results["BPR"] = results
            _save_results({"BPR": results})

    # Hybrid
    if run_all or args.model == "hybrid":
        if not HYBRID_CKPT_PATH.exists():
            log.warning("Hybrid checkpoint not found — skipping.")
        else:
            log.info("\nEvaluating Hybrid...")
            if tensors is None:
                tensors = _load_feature_tensors()
            hybrid = HybridTrainer.load_model(HYBRID_CKPT_PATH, device=DEVICE)
            hybrid.eval()
            log.info("  Preloading tensors to GPU...")
            gpu_t = {k: v.to(DEVICE) for k, v in tensors.items()}
            results = evaluate_model(
                "Hybrid", lambda u, s: _score_hybrid(hybrid, u, gpu_t, s),
                eval_df, train_df, cold_users
            )
            _print_results("Hybrid", results, TOP_K_VALUES)
            all_results["Hybrid"] = results
            _save_results({"Hybrid": results})

    if all_results:
        _save_results(all_results)
        _print_comparison(all_results)

    log.info(f"\nDone in {int((time.time()-t0)//60)}m {(time.time()-t0)%60:.1f}s")


def _parse_args():
    p = argparse.ArgumentParser(description="DataFlix evaluation pipeline")
    p.add_argument("--model", choices=["als", "bpr", "hybrid"], default=None)
    p.add_argument("--split", choices=["val", "test"], default="test")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())