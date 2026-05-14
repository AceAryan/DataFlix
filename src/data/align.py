"""
DataFlix — Step 1: Netflix → MovieLens Alignment
src/data/align.py

Strategy:
  Stage 1 (high confidence): Netflix title+year → IMDB title.basics → ML links.csv (via imdbId)
  Stage 2 (medium confidence): unmatched Netflix → TMDB CSV → ML links.csv (via tmdbId)
  Unmatched after both stages are dropped — no guessing.

Outputs:
  processed/netflix_to_ml_movie_map.json   — {nf_id: {ml_id, imdb_id, match_confidence}}
  processed/alignment_report.csv           — full audit trail for every Netflix movie
"""

import re
import json
import logging
import unicodedata

import pandas as pd
import numpy as np
from pathlib import Path

# Allow running as a standalone script OR imported as a module
try:
    from src.config import (
        NETFLIX_MOVIES_PATH, ML_MOVIES_PATH, ML_LINKS_PATH,
        IMDB_BASICS_PATH, TMDB_CSV_PATH,
        PROCESSED_DIR, NF_TO_ML_MAP_JSON,
    )
except ModuleNotFoundError:
    # Fallback for direct execution: resolve paths relative to project root
    _ROOT = Path(__file__).resolve().parent.parent.parent
    NETFLIX_MOVIES_PATH = _ROOT / "data/raw/netflix/Netflix_Dataset_Movie.csv"
    ML_MOVIES_PATH      = _ROOT / "data/raw/ml-25m/ml-25m/movies.csv"
    ML_LINKS_PATH       = _ROOT / "data/raw/ml-25m/ml-25m/links.csv"
    IMDB_BASICS_PATH    = _ROOT / "data/raw/imdb/title.basics.tsv"
    TMDB_CSV_PATH       = _ROOT / "data/raw/tmdb/TMDB_movie_dataset_v11.csv"
    PROCESSED_DIR       = _ROOT / "data/processed"
    NF_TO_ML_MAP_JSON   = PROCESSED_DIR / "netflix_to_ml_movie_map.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Confidence tier labels (best → worst) ─────────────────────────────────────
EXACT      = "exact"          # title + year match perfectly after normalisation
NORM       = "normalised"     # after stripping articles / punctuation
YEAR_FUZZY = "year_fuzzy"     # normalised title + year ± 1
TMDB_EXACT = "tmdb_exact"     # Stage 2 exact
TMDB_NORM  = "tmdb_normalised"
TMDB_FUZZY = "tmdb_year_fuzzy"

# Confidence ranking (lower = better)
CONFIDENCE_RANK = {
    EXACT: 0, NORM: 1, YEAR_FUZZY: 2,
    TMDB_EXACT: 3, TMDB_NORM: 4, TMDB_FUZZY: 5,
}

# ── Text normalisation ─────────────────────────────────────────────────────────

_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE = re.compile(r"\s+")


def normalise(title: str) -> str:
    """
    Lowercase, strip leading articles, remove punctuation, collapse whitespace.
    Also handles unicode (e.g. accented chars → ASCII).
    """
    if not isinstance(title, str):
        return ""
    # Unicode → ASCII (café → cafe)
    title = unicodedata.normalize("NFKD", title)
    title = title.encode("ascii", "ignore").decode("ascii")
    title = title.lower()
    title = _ARTICLES.sub("", title)
    title = _NON_ALNUM.sub(" ", title)
    title = _MULTI_SPACE.sub(" ", title).strip()
    return title


def extract_ml_year(ml_title: str) -> int | None:
    """Extract year from MovieLens title format: 'Toy Story (1995)' → 1995."""
    m = re.search(r"\((\d{4})\)\s*$", ml_title)
    return int(m.group(1)) if m else None


def strip_ml_year(ml_title: str) -> str:
    """'Toy Story (1995)' → 'Toy Story'."""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", ml_title).strip()


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_netflix_movies() -> pd.DataFrame:
    """
    Returns DataFrame with columns: nf_id (int), nf_title (str), nf_year (Int64).
    Year may be NaN for some entries.
    """
    df = pd.read_csv(NETFLIX_MOVIES_PATH)
    df = df.rename(columns={"Movie_ID": "nf_id", "Name": "nf_title", "Year": "nf_year"})
    df["nf_id"]    = df["nf_id"].astype(int)
    df["nf_year"]  = pd.to_numeric(df["nf_year"], errors="coerce").astype("Int64")
    df["nf_norm"]  = df["nf_title"].map(normalise)
    log.info(f"Loaded {len(df):,} Netflix movies")
    return df


def load_ml_links() -> pd.DataFrame:
    """
    Returns DataFrame: ml_id (int), imdb_id (str 'tt0xxxxxxx'), tmdb_id (Int64).
    """
    df = pd.read_csv(ML_LINKS_PATH)
    df = df.rename(columns={"movieId": "ml_id", "imdbId": "imdb_num", "tmdbId": "tmdb_id"})
    # imdbId in links.csv is numeric (e.g. 114709); convert to 'tt' format
    df["imdb_id"] = df["imdb_num"].apply(
        lambda x: f"tt{int(x):07d}" if pd.notna(x) else None
    )
    df["tmdb_id"] = pd.to_numeric(df["tmdb_id"], errors="coerce").astype("Int64")
    df["ml_id"]   = df["ml_id"].astype(int)
    return df


def load_imdb_basics() -> pd.DataFrame:
    """
    Returns movies-only rows from title.basics with columns:
    tconst, imdb_norm, imdb_year (Int64).
    Filters to titleType == 'movie' to avoid matching TV episodes.
    """
    log.info("Loading IMDB title.basics (this may take ~10s)...")
    df = pd.read_csv(
        IMDB_BASICS_PATH, sep="\t", na_values="\\N", low_memory=False,
        usecols=["tconst", "titleType", "primaryTitle", "startYear"],
    )
    df = df[df["titleType"] == "movie"].copy()
    df["imdb_year"] = pd.to_numeric(df["startYear"], errors="coerce").astype("Int64")
    df["imdb_norm"] = df["primaryTitle"].map(normalise)
    df = df.drop(columns=["titleType", "startYear", "primaryTitle"])
    log.info(f"  {len(df):,} IMDB movie entries loaded")
    return df


def load_tmdb() -> pd.DataFrame:
    """
    Returns TMDB CSV with columns: tmdb_id (Int64), tmdb_norm, tmdb_year (Int64).
    """
    log.info("Loading TMDB CSV...")
    df = pd.read_csv(TMDB_CSV_PATH, low_memory=False,
                     usecols=["id", "title", "release_date"])
    df = df.rename(columns={"id": "tmdb_id"})
    df["tmdb_id"]   = pd.to_numeric(df["tmdb_id"], errors="coerce").astype("Int64")
    df["tmdb_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.astype("Int64")
    df["tmdb_norm"] = df["title"].map(normalise)
    df = df.drop(columns=["title", "release_date"])
    log.info(f"  {len(df):,} TMDB entries loaded")
    return df


# ── Core matching helpers ─────────────────────────────────────────────────────

def _build_imdb_lookup(imdb: pd.DataFrame) -> dict[tuple, list[str]]:
    """
    Returns dict: (norm_title, year) → [tconst, ...]
    Allows O(1) lookup for a given normalised title + year.
    """
    lookup: dict[tuple, list[str]] = {}
    for _, row in imdb.iterrows():
        if pd.isna(row["imdb_year"]):
            continue
        key = (row["imdb_norm"], int(row["imdb_year"]))
        lookup.setdefault(key, []).append(row["tconst"])
    return lookup


def _build_tmdb_lookup(tmdb: pd.DataFrame) -> dict[tuple, list[int]]:
    """Returns dict: (norm_title, year) → [tmdb_id, ...]"""
    lookup: dict[tuple, list[int]] = {}
    for _, row in tmdb.iterrows():
        if pd.isna(row["tmdb_year"]):
            continue
        key = (row["tmdb_norm"], int(row["tmdb_year"]))
        lookup.setdefault(key, []).append(int(row["tmdb_id"]))
    return lookup


def _lookup_imdb(norm: str, year: int | None, imdb_lookup: dict) -> tuple[str | None, str]:
    """
    Try exact → normalised → year±1.
    Returns (tconst, confidence) or (None, "").
    Year parameter is already the Netflix year (Int64 → int or None).
    """
    if year is None:
        return None, ""

    year = int(year)

    # Exact year
    candidates = imdb_lookup.get((norm, year), [])
    if len(candidates) == 1:
        return candidates[0], EXACT
    if len(candidates) > 1:
        # Multiple matches — flag but take first (audited in report)
        return candidates[0], EXACT

    # Year ± 1  (handles US vs international release year discrepancy)
    for delta in [-1, 1]:
        candidates = imdb_lookup.get((norm, year + delta), [])
        if candidates:
            return candidates[0], YEAR_FUZZY

    return None, ""


def _lookup_tmdb(norm: str, year: int | None, tmdb_lookup: dict) -> tuple[int | None, str]:
    """Try exact → year±1 against TMDB. Returns (tmdb_id, confidence)."""
    if year is None:
        return None, ""
    year = int(year)

    candidates = tmdb_lookup.get((norm, year), [])
    if candidates:
        return candidates[0], TMDB_EXACT

    for delta in [-1, 1]:
        candidates = tmdb_lookup.get((norm, year + delta), [])
        if candidates:
            return candidates[0], TMDB_FUZZY

    return None, ""


# ── Main alignment function ───────────────────────────────────────────────────

def build_alignment() -> tuple[dict, pd.DataFrame]:
    """
    Run the full two-stage alignment.

    Returns
    -------
    mapping : dict
        {nf_id (int): {"ml_id": int, "imdb_id": str, "match_confidence": str}}
    report : pd.DataFrame
        Full audit trail for every Netflix movie.
    """
    # ── Load all sources ──
    nf      = load_netflix_movies()
    ml_links = load_ml_links()
    imdb    = load_imdb_basics()
    tmdb    = load_tmdb()

    # ── Pre-build lookup dicts ──
    imdb_lookup = _build_imdb_lookup(imdb)
    tmdb_lookup = _build_tmdb_lookup(tmdb)

    # ── Build reverse maps: tconst → ml_id, tmdb_id → ml_id ──
    tconst_to_ml: dict[str, int] = dict(
        zip(ml_links["imdb_id"], ml_links["ml_id"])
    )
    tmdb_to_ml: dict[int, int] = {
        int(row.tmdb_id): int(row.ml_id)
        for row in ml_links.itertuples()
        if pd.notna(row.tmdb_id)
    }

    # ── Per-movie alignment loop ──
    records = []

    for row in nf.itertuples(index=False):
        nf_id   = row.nf_id
        nf_norm = row.nf_norm
        nf_year = row.nf_year if pd.notna(row.nf_year) else None

        result = {
            "nf_id":            nf_id,
            "nf_title":         row.nf_title,
            "nf_year":          nf_year,
            "ml_id":            None,
            "imdb_id":          None,
            "tmdb_id_matched":  None,
            "match_confidence": "unmatched",
            "stage":            None,
        }

        # ── Stage 1: IMDB bridge ──
        tconst, conf = _lookup_imdb(nf_norm, nf_year, imdb_lookup)
        if tconst and tconst in tconst_to_ml:
            result.update({
                "ml_id":            tconst_to_ml[tconst],
                "imdb_id":          tconst,
                "match_confidence": conf,
                "stage":            1,
            })
            records.append(result)
            continue

        # ── Stage 2: TMDB fallback ──
        tmdb_id, conf = _lookup_tmdb(nf_norm, nf_year, tmdb_lookup)
        if tmdb_id and tmdb_id in tmdb_to_ml:
            result.update({
                "ml_id":            tmdb_to_ml[tmdb_id],
                "tmdb_id_matched":  tmdb_id,
                "match_confidence": conf,
                "stage":            2,
            })
            records.append(result)
            continue

        # ── Unmatched ──
        records.append(result)

    report = pd.DataFrame(records)

    # ── Build mapping dict (matched only) ──
    matched = report[report["ml_id"].notna()].copy()
    mapping: dict[int, dict] = {}
    for row in matched.itertuples(index=False):
        mapping[int(row.nf_id)] = {
            "ml_id":            int(row.ml_id),
            "imdb_id":          row.imdb_id,
            "match_confidence": row.match_confidence,
        }

    # ── Logging summary ──
    total     = len(report)
    n_matched = len(matched)
    n_stage1  = int((report["stage"] == 1).sum())
    n_stage2  = int((report["stage"] == 2).sum())
    n_unmatched = total - n_matched

    log.info("=" * 50)
    log.info(f"Alignment complete: {total:,} Netflix movies")
    log.info(f"  Stage 1 (IMDB)  : {n_stage1:,}  ({n_stage1/total*100:.1f}%)")
    log.info(f"  Stage 2 (TMDB)  : {n_stage2:,}  ({n_stage2/total*100:.1f}%)")
    log.info(f"  Unmatched (drop): {n_unmatched:,}  ({n_unmatched/total*100:.1f}%)")
    log.info(f"  Total matched   : {n_matched:,}  ({n_matched/total*100:.1f}%)")

    conf_counts = report["match_confidence"].value_counts()
    log.info("  Confidence breakdown:")
    for conf, count in conf_counts.items():
        log.info(f"    {conf:<20}: {count:,}")

    return mapping, report


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_alignment(mapping: dict, report: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Main mapping JSON
    with open(NF_TO_ML_MAP_JSON, "w") as f:
        json.dump(mapping, f, indent=2)
    log.info(f"Saved mapping → {NF_TO_ML_MAP_JSON}")

    # Full audit CSV
    report_path = PROCESSED_DIR / "alignment_report.csv"
    report.to_csv(report_path, index=False)
    log.info(f"Saved audit report → {report_path}")


# ── Public entry point (called from pipeline) ─────────────────────────────────

def run_alignment() -> dict:
    """
    Build and save alignment. Returns mapping dict.
    Skips rebuild if mapping already exists (use force=True to override).
    """
    if NF_TO_ML_MAP_JSON.exists():
        log.info(f"Alignment map already exists at {NF_TO_ML_MAP_JSON} — loading.")
        with open(NF_TO_ML_MAP_JSON) as f:
            return {int(k): v for k, v in json.load(f).items()}

    mapping, report = build_alignment()
    save_alignment(mapping, report)
    return mapping


def run_alignment_force() -> dict:
    """Force a full rebuild even if the mapping file already exists."""
    mapping, report = build_alignment()
    save_alignment(mapping, report)
    return mapping


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Netflix → ML alignment")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if mapping already exists")
    args = parser.parse_args()

    if args.force:
        run_alignment_force()
    else:
        run_alignment()