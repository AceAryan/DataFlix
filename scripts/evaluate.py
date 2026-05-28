"""
DataFlix — Evaluation Script
scripts/evaluate.py

Evaluation protocols per model:

  ALS    — RMSE / MAE only (rating prediction baseline)
             ALS predicts ratings, not rankings. No NDCG/Recall.

  BPR    — NDCG@K, Recall@K, MRR@K + RMSE/MAE
             Seen items excluded from ranking (standard implicit feedback).

  Hybrid — NDCG@K, Recall@K, MRR@K + RMSE/MAE
             Same protocol as BPR.

Relevance: test item is relevant if raw rating >= 4.0
Cold users: users with < COLD_EVAL_THRESHOLD training ratings

Usage:
  python scripts/evaluate.py                # all models on test split
  python scripts/evaluate.py --model als
  python scripts/evaluate.py --model bpr
  python scripts/evaluate.py --model hybrid
  python scripts/evaluate.py --split val
"""

import argparse, json, logging, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.config import (
    RESULTS_DIR, DEVICE,
    TEST_CSV, VAL_CSV, TRAIN_CSV,
    COLD_START_THRESHOLD,
    SBERT_EMBEDDINGS_PATH, IMDB_FEATURES_PATH,
    POPULARITY_PATH, HISTORY_EMBEDDINGS_PATH,
    ALS_PATH, BPR_FACTORS_PATH, HYBRID_CKPT_PATH,
    TOP_K_VALUES, EVAL_BATCH_SIZE, RELEVANCE_RATING,
    set_seed,
)
from src.models.als    import ALS
from src.models.bpr    import BPR
from src.models.hybrid import HybridModel, HybridTrainer, score_all_items

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EVAL_REPORT_JSON = RESULTS_DIR / "evaluation_report.json"
EVAL_REPORT_CSV  = RESULTS_DIR / "evaluation_report.csv"
COLD_EVAL_THRESHOLD = max(COLD_START_THRESHOLD, 20)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _dcg(rel, k):
    r = rel[:k].astype(np.float32)
    return float((r / np.log2(np.arange(2, len(r)+2))).sum()) if r.sum() > 0 else 0.0

def ndcg_at_k(ranked, relevant, k):
    if not relevant: return 0.0
    rel   = np.array([1 if i in relevant else 0 for i in ranked[:k]])
    ideal = np.ones(min(len(relevant), k))
    id_   = _dcg(ideal, k)
    return _dcg(rel, k) / id_ if id_ > 0 else 0.0

def recall_at_k(ranked, relevant, k):
    if not relevant: return 0.0
    return sum(1 for i in ranked[:k] if i in relevant) / len(relevant)

def mrr_at_k(ranked, relevant, k):
    for rank, item in enumerate(ranked[:k], 1):
        if item in relevant: return 1.0 / rank
    return 0.0


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_data(split):
    csv = VAL_CSV if split == "val" else TEST_CSV
    log.info(f"Loading {split} split...")
    eval_df  = pd.read_csv(csv)
    train_df = pd.read_csv(TRAIN_CSV)
    counts   = train_df.groupby("user_idx").size()
    cold     = set(counts[counts < COLD_EVAL_THRESHOLD].index.tolist())
    log.info(f"  Eval={len(eval_df):,} | Train={len(train_df):,} | Cold={len(cold):,}")
    return eval_df, train_df, cold


def _build_ground_truth(eval_df, train_df):
    """
    user_relevant : items with raw rating >= RELEVANCE_RATING
    user_seen     : all training items (excluded from ranking)
    user_ratings  : all (movie_idx, rating_centered) for RMSE
    """
    user_relevant: dict[int, set]  = defaultdict(set)
    user_ratings:  dict[int, list] = defaultdict(list)

    for row in eval_df.itertuples(index=False):
        uid, mid = int(row.user_idx), int(row.movie_idx)
        user_ratings[uid].append((mid, float(row.rating_centered)))
        if float(row.rating) >= RELEVANCE_RATING:
            user_relevant[uid].add(mid)

    user_seen: dict[int, set] = defaultdict(set)
    for row in train_df.itertuples(index=False):
        user_seen[int(row.user_idx)].add(int(row.movie_idx))

    n_with = len(user_relevant)
    log.info(f"  Users with >=1 relevant item: {n_with:,} / {len(user_ratings):,}")
    return dict(user_relevant), dict(user_seen), dict(user_ratings)


def _load_tensors():
    return {
        "sbert":   torch.load(SBERT_EMBEDDINGS_PATH,   weights_only=False),
        "imdb":    torch.load(IMDB_FEATURES_PATH,      weights_only=False),
        "pop":     torch.load(POPULARITY_PATH,          weights_only=False),
        "history": torch.load(HISTORY_EMBEDDINGS_PATH, weights_only=False),
    }


# ── ALS: rating eval only ─────────────────────────────────────────────────────

def eval_als(als: ALS, eval_df, train_df, cold) -> dict:
    log.info("  Computing ALS RMSE/MAE...")
    segs  = ["overall","warm","cold"]
    rmse_acc = {s: [] for s in segs}
    mae_acc  = {s: [] for s in segs}

    for uid, grp in eval_df.groupby("user_idx"):
        uid    = int(uid)
        warmth = "cold" if uid in cold else "warm"
        mids   = grp["movie_idx"].values.astype(int)
        true_r = grp["rating_centered"].values.astype(np.float32)
        pred_r = np.array([als.predict(uid, mid) for mid in mids], dtype=np.float32)
        rmse   = float(np.sqrt(np.mean((true_r - pred_r)**2)))
        mae    = float(np.mean(np.abs(true_r - pred_r)))
        for s in ["overall", warmth]:
            rmse_acc[s].append(rmse)
            mae_acc[s].append(mae)

    results = {}
    for s in segs:
        if rmse_acc[s]:
            results[s] = {
                "rmse": float(np.mean(rmse_acc[s])),
                "mae":  float(np.mean(mae_acc[s])),
                "n_users": len(rmse_acc[s]),
                "note": "rating prediction only — not a ranking model",
            }
        else:
            results[s] = {"n_users": 0}
    return results


# ── BPR / Hybrid: ranking + rating eval ──────────────────────────────────────

def eval_ranking(name, score_fn, eval_df, train_df, cold,
                 k_values=TOP_K_VALUES) -> dict:
    user_relevant, user_seen, user_ratings = _build_ground_truth(eval_df, train_df)

    segs  = ["overall","warm","cold"]
    accum = {s: defaultdict(list) for s in segs}

    users   = list(user_relevant.keys())
    n       = len(users)
    log.info(f"  Evaluating {n:,} users...")
    t = time.time()

    for i, uid in enumerate(users):
        if i % 5000 == 0 and i > 0:
            eta = (time.time()-t)/i*(n-i)
            log.info(f"    {i:>6}/{n:,}  ETA {eta/60:.1f}m")

        relevant  = user_relevant[uid]
        seen      = user_seen.get(uid, set())
        warmth    = "cold" if uid in cold else "warm"
        user_segs = ["overall", warmth]

        scores  = score_fn(uid, seen)
        top_max = max(k_values)
        if np.isfinite(scores).sum() < top_max:
            continue

        top_idx = np.argpartition(scores, -top_max)[-top_max:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        for k in k_values:
            for s in user_segs:
                accum[s][f"ndcg@{k}"].append(ndcg_at_k(top_idx, relevant, k))
                accum[s][f"recall@{k}"].append(recall_at_k(top_idx, relevant, k))
                accum[s][f"mrr@{k}"].append(mrr_at_k(top_idx, relevant, k))

        pairs = user_ratings.get(uid, [])
        if pairs:
            true_r = np.array([r for _,r in pairs], dtype=np.float32)
            pred_r = np.array(
                [float(scores[m]) if np.isfinite(scores[m]) else 0.0
                 for m,_ in pairs], dtype=np.float32
            )
            rmse = float(np.sqrt(np.mean((true_r-pred_r)**2)))
            mae  = float(np.mean(np.abs(true_r-pred_r)))
            for s in user_segs:
                accum[s]["rmse"].append(rmse)
                accum[s]["mae"].append(mae)

    results = {}
    for s, metrics in accum.items():
        results[s] = {
            m: float(np.mean(v)) if v else float("nan")
            for m, v in metrics.items()
        }
        results[s]["n_users"] = len(accum[s].get(f"ndcg@{k_values[0]}", []))
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_als(results):
    log.info("\n  ── ALS (rating prediction only) ──")
    log.info(f"  {'Segment':<10}  {'RMSE':<10}  {'MAE':<10}  {'N':<10}")
    log.info("  " + "─"*44)
    for s in ["overall","warm","cold"]:
        m = results.get(s, {})
        if m.get("n_users", 0) == 0: continue
        log.info(f"  {s:<10}  {m.get('rmse',float('nan')):<10.4f}  "
                 f"{m.get('mae',float('nan')):<10.4f}  {m['n_users']:<10,}")


def _print_ranking(name, results, k_values):
    log.info(f"\n  ── {name} ──")
    hdr = f"  {'Segment':<10}" + "".join(
        f"  {'NDCG@'+str(k):<10}{'Recall@'+str(k):<12}{'MRR@'+str(k):<10}"
        for k in k_values
    ) + f"  {'RMSE':<8}{'MAE':<8}{'N':<8}"
    log.info(hdr)
    log.info("  " + "─"*(len(hdr)-2))
    for s in ["overall","warm","cold"]:
        m = results.get(s, {})
        if m.get("n_users", 0) == 0: continue
        row = f"  {s:<10}"
        for k in k_values:
            row += (f"  {m.get(f'ndcg@{k}',float('nan')):<10.4f}"
                    f"{m.get(f'recall@{k}',float('nan')):<12.4f}"
                    f"{m.get(f'mrr@{k}',float('nan')):<10.4f}")
        row += (f"  {m.get('rmse',float('nan')):<8.4f}"
                f"{m.get('mae',float('nan')):<8.4f}"
                f"{m.get('n_users',0):<8,}")
        log.info(row)


def _save(new):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    if EVAL_REPORT_JSON.exists():
        with open(EVAL_REPORT_JSON) as f:
            existing = json.load(f)
    existing.update(new)
    with open(EVAL_REPORT_JSON, "w") as f:
        json.dump(existing, f, indent=2)
    rows = []
    for model, segs in existing.items():
        for seg, metrics in segs.items():
            rows.append({"model": model, "segment": seg, **metrics})
    pd.DataFrame(rows).to_csv(EVAL_REPORT_CSV, index=False)
    log.info(f"  Saved → {EVAL_REPORT_JSON.name}  &  {EVAL_REPORT_CSV.name}")


def _print_comparison(all_results):
    k = 10
    log.info("")
    log.info("╔═══════════════════════════════════════════════════════╗")
    log.info("║  Final Results — overall segment, test split           ║")
    log.info("╠══════════════╦══════════╦══════════╦══════════════════╣")
    log.info("║  Model       ║ NDCG@10  ║ Recall@10║ RMSE             ║")
    log.info("╠══════════════╬══════════╬══════════╬══════════════════╣")
    for name, res in all_results.items():
        m    = res.get("overall", {})
        ndcg = m.get(f"ndcg@{k}", None)
        rec  = m.get(f"recall@{k}", None)
        rmse = m.get("rmse", None)
        if ndcg is None:
            log.info(f"║  {name:<12}║ {'N/A':<8} ║ {'N/A':<8} ║ {rmse:<16.4f} ║")
        else:
            log.info(f"║  {name:<12}║ {ndcg:<8.4f} ║ {rec:<8.4f} ║ {rmse:<16.4f} ║")
    log.info("╚══════════════╩══════════╩══════════╩══════════════════╝")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    set_seed()
    t0 = time.time()
    eval_df, train_df, cold = _load_data(args.split)
    run_all     = args.model is None
    all_results = {}
    tensors     = None

    # ALS
    if run_all or args.model == "als":
        if not ALS_PATH.exists():
            log.warning("ALS factors not found — skipping.")
        else:
            log.info("\nEvaluating ALS...")
            als     = ALS.load(ALS_PATH)
            results = eval_als(als, eval_df, train_df, cold)
            _print_als(results)
            all_results["ALS"] = results
            _save({"ALS": results})

    # BPR
    if run_all or args.model == "bpr":
        if not BPR_FACTORS_PATH.exists():
            log.warning("BPR factors not found — skipping.")
        else:
            log.info("\nEvaluating BPR...")
            bpr = BPR.load(BPR_FACTORS_PATH, device=DEVICE)
            bpr.model.eval()

            def bpr_score(uid, seen):
                with torch.no_grad():
                    s = bpr.model.score_all_items(uid).cpu().numpy()
                if seen:
                    s = s.copy(); s[list(seen)] = -np.inf
                return s

            results = eval_ranking("BPR", bpr_score, eval_df, train_df, cold)
            _print_ranking("BPR", results, TOP_K_VALUES)
            all_results["BPR"] = results
            _save({"BPR": results})

    # Hybrid
    if run_all or args.model == "hybrid":
        if not HYBRID_CKPT_PATH.exists():
            log.warning("Hybrid checkpoint not found — skipping.")
        else:
            log.info("\nEvaluating Hybrid...")
            if tensors is None:
                tensors = _load_tensors()
            hybrid = HybridTrainer.load_model(HYBRID_CKPT_PATH, device=DEVICE)
            hybrid.eval()

            log.info("  Preloading tensors to GPU...")
            gpu = {k: v.to(DEVICE) for k, v in tensors.items()}

            def hybrid_score(uid, seen):
                s = score_all_items(
                    hybrid, uid,
                    gpu["sbert"], gpu["imdb"], gpu["pop"], gpu["history"],
                    device=DEVICE, batch_size=EVAL_BATCH_SIZE,
                )
                if seen:
                    s = s.copy(); s[list(seen)] = -np.inf
                return s

            results = eval_ranking("Hybrid", hybrid_score, eval_df, train_df, cold)
            _print_ranking("Hybrid", results, TOP_K_VALUES)
            all_results["Hybrid"] = results
            _save({"Hybrid": results})

    if all_results:
        _save(all_results)
        _print_comparison(all_results)

    log.info(f"\nDone in {int((time.time()-t0)//60)}m {(time.time()-t0)%60:.1f}s")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["als","bpr","hybrid"], default=None)
    p.add_argument("--split", choices=["val","test"], default="test")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())