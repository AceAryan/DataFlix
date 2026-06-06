"""
DataFlix — Master Evaluation Script
scripts/evaluate.py

Usage:
  python scripts/evaluate.py --models all
  python scripts/evaluate.py --models lightgcn sasrec twotower
  python scripts/evaluate.py --models bpr --split val
  python scripts/evaluate.py --models easr
"""

import argparse, json, logging, sys, time, pickle
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from src.config import (
    RESULTS_DIR, DEVICE, PROCESSED_DIR,
    TEST_CSV, VAL_CSV, TRAIN_CSV, COLD_START_THRESHOLD,
    SBERT_EMBEDDINGS_PATH, IMDB_FEATURES_PATH, POPULARITY_PATH, HISTORY_EMBEDDINGS_PATH,
    ALS_PATH, BPR_FACTORS_PATH, HYBRID_CKPT_PATH, 
    TOP_K_VALUES, EVAL_BATCH_SIZE, RELEVANCE_RATING, LATENT_DIM_K,
    set_seed,
)

from src.models.als      import ALS
from src.models.bpr      import BPR
from src.models.easr     import EASR, score_easr                        # ← EASE^R
from src.models.hybrid   import HybridModel, score_all_items as score_hybrid
from src.models.twotower import TwoTowerModel, score_twotower
from src.models.lightgcn import LightGCN, score_lightgcn
# from src.models.sasrec   import SASRec, score_sasrec

TWOTOWER_CKPT_PATH  = RESULTS_DIR / "twotower_best.pt"
LIGHTGCN_CKPT_PATH  = RESULTS_DIR / "lightgcn_best.pt"
SASREC_CKPT_PATH    = RESULTS_DIR / "sasrec_best.pt"
LIGHTGCN_GRAPH_PATH = PROCESSED_DIR / "lightgcn_graph.npz"
EASR_PATH           = RESULTS_DIR / "easr.npz"                         # ← EASE^R checkpoint

EVAL_REPORT_JSON = RESULTS_DIR / "evaluation_report.json"
COLD_EVAL_THRESHOLD = max(COLD_START_THRESHOLD, 20)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

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

# ── Data Helpers ──────────────────────────────────────────────────────────────

def _load_data(split):
    csv = VAL_CSV if split == "val" else TEST_CSV
    log.info(f"Loading {split} split...")
    eval_df  = pd.read_csv(csv)
    train_df = pd.read_csv(TRAIN_CSV)
    
    counts = train_df.groupby("user_idx").size()
    cold = set(counts[counts < COLD_EVAL_THRESHOLD].index.tolist())
    log.info(f"  Eval={len(eval_df):,} | Train={len(train_df):,} | Cold={len(cold):,}")
    
    return eval_df, train_df, cold

def _build_ground_truth(eval_df, train_df):
    user_relevant = defaultdict(set)
    user_seen = defaultdict(set)

    for row in eval_df.itertuples(index=False):
        if float(row.rating) >= RELEVANCE_RATING:
            user_relevant[int(row.user_idx)].add(int(row.movie_idx))

    for row in train_df.itertuples(index=False):
        user_seen[int(row.user_idx)].add(int(row.movie_idx))

    return dict(user_relevant), dict(user_seen)

def _get_sasrec_histories(train_df):
    if "timestamp" in train_df.columns:
        train_df = train_df.sort_values(["user_idx", "timestamp"])
    return train_df.groupby("user_idx")["movie_idx"].apply(list).to_dict()

# ── Universal Ranking Evaluator ───────────────────────────────────────────────

def evaluate_model(name, score_fn, eval_df, train_df, cold, k_values=TOP_K_VALUES):
    user_relevant, user_seen = _build_ground_truth(eval_df, train_df)
    segs = ["overall", "warm", "cold"]
    accum = {s: defaultdict(list) for s in segs}

    users = list(user_relevant.keys())
    n = len(users)
    log.info(f"  Evaluating {n:,} users for {name}...")
    t0 = time.time()

    for i, uid in enumerate(users):
        if i > 0 and i % 5000 == 0:
            eta = (time.time()-t0)/i*(n-i)
            log.info(f"    {i:>6}/{n:,}  ETA {eta/60:.1f}m")

        relevant = user_relevant[uid]
        seen = user_seen.get(uid, set())
        warmth = "cold" if uid in cold else "warm"

        scores = score_fn(uid)
        
        if seen:
            scores[list(seen)] = -np.inf

        top_max = max(k_values)
        if np.isfinite(scores).sum() < top_max:
            continue

        top_idx = np.argpartition(scores, -top_max)[-top_max:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        for k in k_values:
            for s in ["overall", warmth]:
                accum[s][f"ndcg@{k}"].append(ndcg_at_k(top_idx, relevant, k))
                accum[s][f"recall@{k}"].append(recall_at_k(top_idx, relevant, k))

    results = {}
    for s, metrics in accum.items():
        results[s] = {m: float(np.mean(v)) if v else float("nan") for m, v in metrics.items()}
        results[s]["n_users"] = len(accum[s].get(f"ndcg@{k_values[0]}", []))
    return results

def _print_comparison(all_results):
    k = 10
    log.info(f"\n╔{'═'*65}╗")
    log.info(f"║  Final Ranking Results — (NDCG@{k} and Recall@{k}){'':>19}║")
    log.info(f"╠{'═'*15}╦{'═'*24}╦{'═'*24}╣")
    log.info(f"║  Model        ║  OVERALL Seg.          ║  COLD Seg.             ║")
    log.info(f"║               ║  NDCG    | Recall      ║  NDCG    | Recall      ║")
    log.info(f"╠{'═'*15}╬{'═'*24}╬{'═'*24}╣")
    
    for name, res in all_results.items():
        ov = res.get("overall", {})
        co = res.get("cold", {})
        
        o_n = ov.get(f"ndcg@{k}", 0.0); o_r = ov.get(f"recall@{k}", 0.0)
        c_n = co.get(f"ndcg@{k}", 0.0); c_r = co.get(f"recall@{k}", 0.0)
        
        log.info(f"║  {name:<12} ║  {o_n:.4f}  | {o_r:.4f}      ║  {c_n:.4f}  | {c_r:.4f}      ║")
    log.info(f"╚{'═'*15}╩{'═'*24}╩{'═'*24}╝")

# ── Incremental Save Helper ───────────────────────────────────────────────────

def _save_result(name: str, result: dict) -> None:
    """
    Merge a single model's result into the JSON report immediately after eval.
    - Reads the current file (if any) so nothing else is lost.
    - Overwrites the entry for `name` with the fresh result.
    - Writes back atomically via a temp file to avoid corruption on crash.
    """
    existing: dict = {}
    if EVAL_REPORT_JSON.exists():
        try:
            with open(EVAL_REPORT_JSON, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            log.warning("  Could not read existing report — starting fresh.")

    existing[name] = result  # upsert: add new or overwrite stale

    tmp = EVAL_REPORT_JSON.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(existing, f, indent=2)
    tmp.replace(EVAL_REPORT_JSON)   # atomic on POSIX; best-effort on Windows

    log.info(f"  ✓ Saved '{name}' → {EVAL_REPORT_JSON.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    set_seed()
    eval_df, train_df, cold = _load_data(args.split)
    all_results = {}
    
    to_run = set(args.models)
    if "all" in to_run:
        to_run = {"als", "bpr", "easr", "hybrid", "twotower", "lightgcn", "sasrec"}

    # 1. ALS
    if "als" in to_run and ALS_PATH.exists():
        log.info("\nEvaluating ALS...")
        als = ALS.load(ALS_PATH)
        def als_score(uid): return als.score_all_items(uid)  # BUG 4 FIX: includes global_mean + biases
        result = evaluate_model("ALS", als_score, eval_df, train_df, cold)
        all_results["ALS"] = result
        _save_result("ALS", result)

    # 2. BPR
    if "bpr" in to_run and BPR_FACTORS_PATH.exists():
        log.info("\nEvaluating BPR...")
        bpr = BPR.load(BPR_FACTORS_PATH, device=DEVICE)
        bpr.model.eval()
        def bpr_score(uid):
            with torch.no_grad(): return bpr.model.score_all_items(uid).cpu().numpy()
        result = evaluate_model("BPR", bpr_score, eval_df, train_df, cold)
        all_results["BPR"] = result
        _save_result("BPR", result)

    # 3. EASE^R  ── pure numpy, no GPU needed
    if "easr" in to_run and EASR_PATH.exists():
        log.info("\nEvaluating EASE^R...")
        easr = EASR.load(EASR_PATH)
        def easr_score_fn(uid): return score_easr(easr, uid)
        result = evaluate_model("EASR", easr_score_fn, eval_df, train_df, cold)
        all_results["EASR"] = result
        _save_result("EASR", result)

    # 4. Hybrid & Two-Tower (Need Tensors)
    if ("hybrid" in to_run) or ("twotower" in to_run):
        tensors = {k: torch.load(p, map_location=DEVICE, weights_only=False) for k, p in 
                   [("sbert", SBERT_EMBEDDINGS_PATH), ("imdb", IMDB_FEATURES_PATH), 
                    ("pop", POPULARITY_PATH), ("history", HISTORY_EMBEDDINGS_PATH)]}
        
        if "hybrid" in to_run and HYBRID_CKPT_PATH.exists():
            log.info("\nEvaluating Hybrid...")
            ck = torch.load(HYBRID_CKPT_PATH, map_location=DEVICE, weights_only=False)
            hybrid = HybridModel(n_users=ck["n_users"], n_items=ck["n_items"], embed_dim=ck["embed_dim"])
            hybrid.load_state_dict(ck["model_state"])
            hybrid.to(DEVICE).eval()
            
            def hybrid_score(uid): return score_hybrid(hybrid, uid, tensors["sbert"], tensors["imdb"], tensors["pop"], tensors["history"], device=DEVICE)
            result = evaluate_model("Hybrid", hybrid_score, eval_df, train_df, cold)
            all_results["Hybrid"] = result
            _save_result("Hybrid", result)

        if "twotower" in to_run and TWOTOWER_CKPT_PATH.exists():
            log.info("\nEvaluating Two-Tower...")
            ck = torch.load(TWOTOWER_CKPT_PATH, map_location=DEVICE, weights_only=False)
            tt = TwoTowerModel(n_users=ck["n_users"], n_items=ck["n_items"], embed_dim=ck["embed_dim"])
            tt.load_state_dict(ck["model_state"])
            tt.to(DEVICE).eval()
            
            def tt_score(uid): return score_twotower(tt, uid, tensors["sbert"], tensors["imdb"], tensors["pop"], tensors["history"], device=DEVICE)
            result = evaluate_model("TwoTower", tt_score, eval_df, train_df, cold)
            all_results["TwoTower"] = result
            _save_result("TwoTower", result)

    # 5. LightGCN
    if "lightgcn" in to_run and LIGHTGCN_CKPT_PATH.exists():
        log.info("\nEvaluating LightGCN...")
        ck = torch.load(LIGHTGCN_CKPT_PATH, map_location=DEVICE, weights_only=False)
        lg = LightGCN(ck["n_users"], ck["n_items"], ck["embed_dim"], ck["n_layers"])
        lg.load_state_dict(ck["model_state"])
        lg.to(DEVICE).eval()
        
        norm_adj = sp.load_npz(LIGHTGCN_GRAPH_PATH).tocoo()
        indices = torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col)))
        edge_index = torch.sparse_coo_tensor(indices, torch.FloatTensor(norm_adj.data), torch.Size(norm_adj.shape)).cpu()
        
        def lg_score(uid): return score_lightgcn(lg, uid, edge_index, device=DEVICE)
        result = evaluate_model("LightGCN", lg_score, eval_df, train_df, cold)
        all_results["LightGCN"] = result
        _save_result("LightGCN", result)

    # # 6. SASRec
    # if "sasrec" in to_run and SASREC_CKPT_PATH.exists():
    #     log.info("\nEvaluating SASRec...")
    #     ck = torch.load(SASREC_CKPT_PATH, map_location=DEVICE, weights_only=False)
    #     sr = SASRec(ck["n_items"])
    #     sr.load_state_dict(ck["model_state"])
    #     sr.to(DEVICE).eval()
        
    #     user_histories = _get_sasrec_histories(train_df)
        
    #     def sr_score(uid):
    #         hist = user_histories.get(uid, [])
    #         if not hist: return np.zeros(ck["n_items"], dtype=np.float32)
    #         return score_sasrec(sr, hist, device=DEVICE)
            
    #     result = evaluate_model("SASRec", sr_score, eval_df, train_df, cold)
    #     all_results["SASRec"] = result
    #     _save_result("SASRec", result)

    if all_results:
        _print_comparison(all_results)

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["all"])
    p.add_argument("--split", choices=["val", "test"], default="test")
    return p.parse_args()

if __name__ == "__main__":
    main(_parse_args())