# DataFlix — Hybrid Movie Recommendation System

A hybrid recommendation system built on the Netflix and MovieLens datasets, combining matrix factorization, semantic embeddings, and ranking-aware learning.

Initially developed at IIT Gandhinagar for CS 328 (Intro to Data Science) by Aryan Kumar, Sura Sravan Kumar, and Shreyash Pandit. Extended and maintained by Aryan Kumar.

---

## What it does

- Predicts user movie preferences using collaborative + content-based filtering
- Handles cold-start users via SBERT plot embeddings and genre metadata
- Evaluates in a zero-shot cross-domain setup: trained on MovieLens 25M, evaluated on Netflix
- Optimizes ranking quality with Bayesian Personalized Ranking (BPR)

---

## How it works

**Matrix Factorization** approximates the user-item rating matrix:

```
r̂_ui = μ + b_u + b_i + pᵤᵀ qᵢ
```

**Movie embeddings** combine:
- Latent MF vector
- Genre encoding
- SBERT plot summary embedding
- Popularity score

**User embeddings** combine:
- Latent MF vector
- Aggregated embeddings of liked movies
- Behavioral features (mean rating, variance, activity)

**Ranking loss (BPR):**
```
L = -log σ(r̂_ui − r̂_uj)
```

**Zero-shot cross-domain setup:**  
The model trains on MovieLens 25M and evaluates on Netflix. A TF-IDF fuzzy matcher (`src/data/alignment.py`) aligns ~55% of the Netflix catalog to MovieLens IDs. All Netflix users are cold-start — the model relies entirely on content signals, no learned user factors.

---

## Stack

| Area | Tools |
|---|---|
| Core | Python, PyTorch, NumPy, SciPy |
| NLP | Sentence-BERT (SBERT) |
| Hyperparameter tuning | Optuna |
| Visualization | Matplotlib, Seaborn, UMAP |

---

## Datasets

| Dataset | Size |
|---|---|
| Netflix ratings | ~17M ratings, ~480K users, ~17K movies |
| MovieLens 25M | 25M ratings, training only |
| IMDb / TMDb | Metadata and plot summaries |

---

## Evaluation

| Metric | Measures |
|---|---|
| RMSE, MAE | Rating prediction accuracy |
| Precision@K, Recall@K | Top-K recommendation quality |
| NDCG@K | Ranking quality |
| Coverage | Recommendation diversity |

---

## Running the pipeline

```bash
# 1. Align Netflix → MovieLens movie IDs
python src/data/alignment.py

# 2. Preprocess
python scripts/run_preprocess.py

# 3. Train
python scripts/run_train.py

# 4. Evaluate
python scripts/run_evaluate.py
```

---

## Setup

```bash
git clone https://github.com/AceAryan/dataflix.git
cd dataflix
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

> Raw datasets are not included. Download links and instructions in `data/README.md`.

---

## Project structure

```
dataflix/
├── data/
│   ├── raw/          # Downloaded datasets (not committed)
│   └── processed/    # Preprocessed matrices and tensors (not committed)
├── src/              # Model classes, training logic, data alignment
├── scripts/          # Pipeline scripts (preprocess, train, evaluate)
├── tests/            # GPU checks, model load verification
├── results/          # Training curves, UMAP plots (not committed)
└── reports/          # Project proposal and analysis
```

---

## License

Academic and research use only.