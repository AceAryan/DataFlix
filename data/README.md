# DataFlix Data Setup

Welcome to the DataFlix data preparation guide. The system relies on multiple datasets for collaborative filtering and semantic embeddings. 

> [!WARNING]
> **Raw datasets are NOT committed to this repository** due to size constraints. You must download them manually using the instructions below.

---

## Folder Structure
Before running any scripts, ensure your `data/` directory looks exactly like this:

```text
data/
├── raw/                 ← Create this directory manually
│   ├── imdb/            ← IMDb metadata files
│   ├── ml-32m/          ← MovieLens 32M dataset
│   └── tmdb/            ← TMDb metadata
└── processed/           ← Auto-generated, do not add manually
```

---

## Download instructions

### 1. MovieLens 32M
The core dataset used for training the model.
1. Go to [GroupLens Datasets](https://grouplens.org/datasets/movielens/32m/).
2. Download `ml-32m.zip` and extract it into `data/raw/ml-32m/`.

> [!IMPORTANT]
> **Required Files:** `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`

### 2. TMDb Metadata
Used for fetching plot summaries to generate Sentence-BERT embeddings.
1. Go to the [Full TMDB Movies Dataset 2024](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies).
2. Download and extract it into `data/raw/tmdb/`.

> [!IMPORTANT]
> **Required Files:** `TMDB_movie_dataset_v11.csv`

### 3. IMDb Metadata
Used for fetching additional metadata.
1. Go to [IMDb Datasets](https://datasets.imdbws.com/).
2. Download `title.basics.tsv.gz` and `title.ratings.tsv.gz`.
3. Extract both files and place the unzipped `.tsv` files into `data/raw/imdb/`.

> [!IMPORTANT]
> **Required Files:** `title.basics.tsv`, `title.ratings.tsv`

---

## Generating Processed Data

Once all raw datasets are downloaded and placed in their respective folders, you are ready to process them.

> [!NOTE]
> The processing step generates necessary matrices and tensors for the PyTorch model. These files are saved to `data/processed/` automatically.

Run the following commands from the root directory:

```bash
# 1. Build matrices, graphs and tensors
python scripts/preprocess.py
```

> [!NOTE]
> The `data/processed/` directory is in `.gitignore` — do not commit the processed files to version control.