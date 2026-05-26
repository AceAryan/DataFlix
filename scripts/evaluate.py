"""
DataFlix — Evaluation Script
scripts/evaluate.py

Model evaluation roles:
  ALS    — rating prediction baseline only (RMSE/MAE)
             ALS is explicit feedback MF trained to predict ratings.
             It is NOT evaluated on ranking metrics — ranking unseen items
             is not what ALS was designed for. Its value is as a baseline
             for rating accuracy and as a warm-start for BPR/Hybrid.

  BPR    — ranking model (NDCG, Recall, MRR + RMSE)
             Trained with pairwise ranking loss. Evaluated on ranking
             metrics with seen-item exclusion (standard implicit feedback
             evaluation protocol).

  Hybrid — ranking + content model (NDCG, Recall, MRR + RMSE)
             Trained with BPR loss + content features. Same evaluation
             protocol as BPR. Should outperform BPR especially on cold
             users where content features compensate for sparse CF signal.

Relevant items definition:
  A test item is relevant if raw rating >= 4.0 (user genuinely liked it).
  Using centred ratings as relevance signal was incorrect — a 4-star rating
  from a generous user (mean=4.5) would be centred to -0.5 and excluded.

Usage:
  python scripts/evaluate.py                    # all models
  python scripts/evaluate.py --model als        # ALS rating metrics only
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

EVAL_REPORT_JSON = RESULTS_DIR / "evaluation_report.json"
EVAL_REPORT_CSV  = RESULTS_DIR / "evaluation_report.csv"

# Rating threshold for relevance — items rated >= this are "liked"
RELEVANCE_THRESHOLD = 4.0

# Cold threshold — users with fewer train ratings than this are "cold"
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

    train_counts = train_df.groupby("user_idx").size()
    cold_users   = set(train_counts[train_counts < COLD_EVAL_THRESHOLD].index.tolist())
    log.info(f"  Eval  : {len(eval_df):,} | Train: {len(train_df):,} | "
             f"Cold users: {len(cold_users):,}")
    return eval_df, train_df, cold_users


def _build_ground_truth(eval_df, train_df):
    """
    user_relevant : items rated >= 4.0 in test — positive signal for ranking
    user_seen     : all training items — excluded from ranking recommendations
    user_ratings  : all (movie_idx, rating_centered) — for RMSE computation
    """
    user_relevant: dict[int, set]  = defaultdict(set)
    user_ratings:  dict[int, list] = defaultdict(list)

    for row in eval_df.itertuples(index=False):
        uid = int(row.user_idx)
        mid = int(row.movie_idx)
        user_ratings[uid].append((mid, float(row.rating_centered)))
        if float(row.rating) >= RELEVANCE_THRESHOLD:
            user_relevant[uid].add(mid)

    user_seen: dict[int, set] = defaultdict(set)
    for row in train_df.itertuples(index=False):
        user_seen[int(row.user_idx)].add(int(row.movie_idx))

    n_with = len(user_relevant)
    n_all  = len(user_ratings)
    log.info(f"  Users with >=1 relevant test item: {n_with:,} / {n_all:,}")
    return dict(user_relevant), dict(user_seen), dict(user_ratings)


def _load_feature_tensors():
    return {
        "sbert":   torch.load(SBERT_EMBEDDINGS_PATH,   weights_only=False),
        "imdb":    torch.load(IMDB_FEATURES_PATH,      weights_only=False),
        "pop":     torch.load(POPULARITY_PATH,          weights_only=False),
        "history": torch.load(HISTORY_EMBEDDINGS_PATH, weights_only=False),
    }


# ── Scorers ───────────────────────────────────────────────────────────────────

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


# ── ALS rating evaluation (no ranking) ───────────────────────────────────────

def evaluate_als_ratings(als: ALS, eval_df: pd.DataFrame,
                         train_df: pd.DataFrame, cold_users: set) -> dict:
    """
    Evaluate ALS on rating prediction only (RMSE and MAE).
    ALS is an explicit feedback model — it predicts ratings, not rankings.
    Ranking metrics are not reported for ALS.
    """
    log.info("  Computing ALS rating metrics (RMSE/MAE only)...")

    # Build user_id lookup for segment detection
    uid_lookup = (
        eval_df[["user_idx", "user_id"]].drop_duplicates("user_idx")
        .set_index("user_idx")["user_id"].to_dict()
    )

    segments = ["overall", "warm", "cold"]
    rmse_acc = {seg: [] for seg in segments}
    mae_acc  = {seg: [] for seg in segments}
    n_acc    = {seg: 0  for seg in segments}

    grouped = eval_df.groupby("user_idx")
    for user_idx, group in grouped:
        user_idx = int(user_idx)
        warmth   = "cold" if user_idx in cold_users else "warm"
        segs     = ["overall", warmth]

        movie_ids = group["movie_idx"].values.astype(int)
        true_rc   = group["rating_centered"].values.astype(np.float32)

        # ALS predicts mean-centred ratings directly
        pred_rc = np.array([
            als.predict(user_idx, mid) for mid in movie_ids
        ], dtype=np.float32)

        rmse = float(np.sqrt(np.mean((true_rc - pred_rc) ** 2)))
        mae  = float(np.mean(np.abs(true_rc - pred_rc)))

        for seg in segs:
            rmse_acc[seg].append(rmse)
            mae_acc[seg].append(mae)
            n_acc[seg] += 1

    results = {}
    for seg in segments:
        if rmse_acc[seg]:
            results[seg] = {
                "rmse":    float(np.mean(rmse_acc[seg])),
                "mae":     float(np.mean(mae_acc[seg])),
                "n_users": n_acc[seg],
                "note":    "ALS evaluated on rating prediction only — not a ranking model",
            }
        else:
            results[seg] = {"n_users": 0}

    return results


# ── Ranking evaluation (BPR + Hybrid) ────────────────────────────────────────

def evaluate_ranking(
    model_name: str,
    score_fn,
    eval_df:    pd.DataFrame,
    train_df:   pd.DataFrame,
    cold_users: set,
    k_values:   list = TOP_K_VALUES,
) -> dict:
    """
    Full ranking + rating evaluation for BPR and Hybrid.
    Uses seen-item exclusion (standard implicit feedback protocol).
    Only users with >= 1 relevant test item contribute to ranking metrics.
    """
    user_relevant, user_seen, user_ratings = _build_ground_truth(eval_df, train_df)

    uid_lookup = (
        eval_df[["user_idx", "user_id"]].drop_duplicates("user_idx")
        .set_index("user_idx")["user_id"].to_dict()
    )

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

        # Rating metrics
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

def _print_als(results: dict) -> None:
    log.info("\n  ── ALS (rating prediction baseline) ──")
    log.info(f"  {'Segment':<10}  {'RMSE':<10}  {'MAE':<10}  {'N':<10}")
    log.info("  " + "─" * 44)
    for seg in ["overall", "warm", "cold"]:
        m = results.get(seg, {})
        if not m or m.get("n_users", 0) == 0:
            continue
        log.info(f"  {seg:<10}  {m.get('rmse', float('nan')):<10.4f}  "
                 f"{m.get('mae', float('nan')):<10.4f}  {m.get('n_users', 0):<10,}")
    log.info("  (Ranking metrics not reported for ALS — see module docstring)")


def _print_ranking(name: str, results: dict, k_values: list) -> None:
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


def _save_results(new_results: dict) -> None:
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


def _print_comparison(all_results: dict) -> None:
    k = 10
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  Model Comparison — overall, test split                       ║")
    log.info("╠══════════════╦══════════╦══════════╦════════╦════════════════╣")
    log.info("║  Model       ║ NDCG@10  ║ Recall@10║ RMSE   ║ Notes          ║")
    log.info("╠══════════════╬══════════╬══════════╬════════╬════════════════╣")
    for name, res in all_results.items():
        m = res.get("overall", {})
        ndcg   = m.get(f"ndcg@{k}", None)
        recall = m.get(f"recall@{k}", None)
        rmse   = m.get("rmse", None)
        note   = "rating only" if ndcg is None else ""
        log.info(
            f"║  {name:<12}║ "
            f"{'N/A':<8} ║ " if ndcg is None else
            f"║  {name:<12}║ {ndcg:<8.4f} ║ {recall:<8.4f} ║ "
            f"{rmse:<6.4f} ║ {note:<14} ║"
        )
    log.info("╚══════════════╩══════════╩══════════╩════════╩════════════════╝")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    set_seed()
    t0 = time.time()

    eval_df, train_df, cold_users = _load_eval_data(args.split)
    run_all     = args.model is None
    all_results = {}
    tensors     = None

    # ALS — rating metrics only
    if run_all or args.model == "als":
        if not ALS_PATH.exists():
            log.warning("ALS factors not found — skipping.")
        else:
            log.info("\nEvaluating ALS (rating prediction)...")
            als     = ALS.load(ALS_PATH)
            results = evaluate_als_ratings(als, eval_df, train_df, cold_users)
            _print_als(results)
            all_results["ALS"] = results
            _save_results({"ALS": results})

    # BPR — full ranking + rating
    if run_all or args.model == "bpr":
        if not BPR_FACTORS_PATH.exists():
            log.warning("BPR factors not found — skipping.")
        else:
            log.info("\nEvaluating BPR...")
            bpr = BPR.load(BPR_FACTORS_PATH, device=DEVICE)
            bpr.model.eval()
            results = evaluate_ranking(
                "BPR",
                lambda u, s: _score_bpr(bpr, u, s),
                eval_df, train_df, cold_users,
            )
            _print_ranking("BPR", results, TOP_K_VALUES)
            all_results["BPR"] = results
            _save_results({"BPR": results})

    # Hybrid — full ranking + rating
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
            results = evaluate_ranking(
                "Hybrid",
                lambda u, s: _score_hybrid(hybrid, u, gpu_t, s),
                eval_df, train_df, cold_users,
            )
            _print_ranking("Hybrid", results, TOP_K_VALUES)
            all_results["Hybrid"] = results
            _save_results({"Hybrid": results})

    if all_results:
        _save_results(all_results)
        _print_comparison(all_results)

    log.info(f"\nDone in {int((time.time()-t0)//60)}m {(time.time()-t0)%60:.1f}s")


def _parse_args():
    p = argparse.ArgumentParser(description="DataFlix evaluation")
    p.add_argument("--model", choices=["als", "bpr", "hybrid"], default=None)
    p.add_argument("--split", choices=["val", "test"], default="test")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())