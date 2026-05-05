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
│   ├── ml-25m/          ← MovieLens 25M dataset
│   ├── netflix/         ← Netflix prize dataset
│   └── tmdb/            ← TMDb metadata
└── processed/           ← Auto-generated, do not add manually
```

---

## Download instructions

### 1. MovieLens 25M
The core dataset used for training the model.
1. Go to [GroupLens Datasets](https://grouplens.org/datasets/movielens/25m/).
2. Download `ml-25m.zip` and extract it into `data/raw/ml-25m/`.

> [!IMPORTANT]
> **Required Files:** `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`

### 2. Netflix Ratings Dataset
Used for the zero-shot cross-domain evaluation.
1. Go to the [Kaggle Netflix Ratings Dataset](https://www.kaggle.com/datasets/rishitjavia/netflix-movie-rating-dataset).
2. Download and extract the archive into `data/raw/netflix/`.

> [!IMPORTANT]
> **Required Files:** `Netflix_Dataset_Movie.csv`, `Netflix_Dataset_Rating.csv`

### 3. TMDb Metadata
Used for fetching plot summaries to generate Sentence-BERT embeddings.
1. Go to the [Full TMDB Movies Dataset 2024](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies).
2. Download and extract it into `data/raw/tmdb/`.

> [!IMPORTANT]
> **Required Files:** `TMDB_movie_dataset_v11.csv`

### 4. IMDb Metadata
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
# 1. Align Netflix IDs to MovieLens IDs
python src/data/alignment.py

# 2. Build matrices and tensors
python scripts/run_preprocess.py
```

> [!NOTE]
> The `data/processed/` directory is in `.gitignore` — do not commit the processed files to version control.