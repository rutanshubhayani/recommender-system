import argparse
import json
import zipfile
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
import requests
from joblib import dump, load
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path("data")
ART_DIR = Path("artifacts")
ML_DIR = DATA_DIR / "ml-latest-small"
RATINGS_CSV = ML_DIR / "ratings.csv"
MOVIES_CSV = ML_DIR / "movies.csv"


def download_movielens():
    ML_DIR.mkdir(parents=True, exist_ok=True)

    # 1) If CSVs are already there, skip
    if RATINGS_CSV.exists() and MOVIES_CSV.exists():
        print("[i] MovieLens already present.")
        return

    # 2) If a local zip exists, use it instead of downloading
    local_zip = DATA_DIR / "ml-latest-small.zip"
    if local_zip.exists():
        print("[i] Found local zip. Extracting…")
        with zipfile.ZipFile(local_zip, "r") as zf:
            zf.extractall(DATA_DIR)
        print("[i] Done.")
        return

    # 3) Otherwise download using certifi CA bundle
    print("[i] Downloading MovieLens (ml-latest-small)…")
    r = requests.get(DATA_URL, stream=True, timeout=60, verify=certifi.where())
    r.raise_for_status()
    zpath = local_zip
    with open(zpath, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("[i] Extracting…")
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(DATA_DIR)
    # keep the zip for reuse
    print("[i] Done.")


def load_data():
    df_ratings = pd.read_csv(RATINGS_CSV)
    df_movies = pd.read_csv(MOVIES_CSV)
    return df_ratings, df_movies


def build_content_based(df_movies: pd.DataFrame):
    ART_DIR.mkdir(exist_ok=True, parents=True)
    print("[i] Building TF-IDF for content-based similarity…")
    # Combine title + genres; fill NA
    text = (
        df_movies["title"].fillna("")
        + " "
        + df_movies["genres"].fillna("").str.replace("|", " ", regex=False)
    )
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
    X = vectorizer.fit_transform(text)
    # store cosine similarities as a sparse matrix to save memory
    print("[i] Computing cosine similarity (this may take ~10s)…")
    cos = cosine_similarity(X, dense_output=False)
    save_npz(ART_DIR / "tfidf_cosine.npz", csr_matrix(cos))
    dump(vectorizer, ART_DIR / "tfidf_vectorizer.pkl")
    print("[i] Content-based artifacts saved.")


def build_collaborative(df_ratings: pd.DataFrame, df_movies: pd.DataFrame, n_components=60):
    ART_DIR.mkdir(exist_ok=True, parents=True)
    print("[i] Building user–item matrix…")
    # Map ids to contiguous indices
    user_ids = sorted(df_ratings["userId"].unique())
    movie_ids = sorted(df_movies["movieId"].unique())
    user_to_idx = {int(u): int(i) for i, u in enumerate(user_ids)}
    movie_to_idx = {int(m): int(i) for i, m in enumerate(movie_ids)}

    rows = df_ratings["userId"].map(user_to_idx).values
    cols = df_ratings["movieId"].map(movie_to_idx).values
    vals = df_ratings["rating"].astype(np.float32).values
    UI = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
        dtype=np.float32,
    )

    # Mean-center per user (only on rated entries)
    print("[i] Mean-centering by user…")
    user_means = np.zeros(UI.shape[0], dtype=np.float32)
    UI_c = UI.copy().tocsr()
    for u in tqdm(range(UI_c.shape[0]), desc="users"):
        start, end = UI_c.indptr[u], UI_c.indptr[u + 1]
        if end > start:
            mean_u = float(UI_c.data[start:end].mean())
            user_means[u] = mean_u
            UI_c.data[start:end] -= mean_u
        else:
            user_means[u] = 0.0

    # Truncated SVD (matrix factorization)
    print("[i] Running Truncated SVD (matrix factorization)…")
    svd = TruncatedSVD(
        n_components=min(n_components, min(UI_c.shape) - 1), random_state=42
    )
    U = svd.fit_transform(UI_c)  # (n_users, k)
    S = svd.singular_values_  # (k,)
    Vt = svd.components_  # (k, n_movies)

    # Save artifacts
    np.save(ART_DIR / "svd_U.npy", U.astype(np.float32))
    np.save(ART_DIR / "svd_S.npy", S.astype(np.float32))
    np.save(ART_DIR / "svd_Vt.npy", Vt.astype(np.float32))
    np.save(ART_DIR / "user_means.npy", user_means)

    # ✅ Cast NumPy int64 keys/values to plain int for JSON
    user_to_idx_json = {int(k): int(v) for k, v in user_to_idx.items()}
    movie_to_idx_json = {int(k): int(v) for k, v in movie_to_idx.items()}

    with open(ART_DIR / "user_index.json", "w") as f:
        json.dump(user_to_idx_json, f)
    with open(ART_DIR / "movie_index.json", "w") as f:
        json.dump(movie_to_idx_json, f)

    print("[i] Collaborative artifacts saved.")


def ensure_built():
    if not (ART_DIR / "tfidf_cosine.npz").exists():
        _, df_movies = load_data()
        build_content_based(df_movies)
    if not (ART_DIR / "svd_U.npy").exists():
        df_ratings, df_movies = load_data()
        build_collaborative(df_ratings, df_movies)


def search_title(df_movies, query, topn=10):
    q = query.lower()
    hits = df_movies[df_movies["title"].str.lower().str.contains(q, na=False)]
    return hits.head(topn)[["movieId", "title", "genres"]]


def recommend_similar(df_movies, title, topn=10):
    cos = load_npz(ART_DIR / "tfidf_cosine.npz")
    _ = load(ART_DIR / "tfidf_vectorizer.pkl")  # kept for completeness; not used directly

    # find exact title
    matches = df_movies.index[
        df_movies["title"].str.lower() == title.lower()
    ].tolist()
    if not matches:
        # fallback to contains
        contains = df_movies.index[
            df_movies["title"].str.lower().str.contains(title.lower(), na=False)
        ].tolist()
        if not contains:
            raise ValueError(f"Movie '{title}' not found. Try --search \"{title}\"")
        idx = contains[0]
        print(f"[!] Using closest match: {df_movies.iloc[idx]['title']}")
    else:
        idx = matches[0]

    sims = cos[idx].toarray().ravel()
    # Exclude itself and sort
    sims[idx] = -1
    top_idx = np.argsort(-sims)[:topn]
    out = df_movies.iloc[top_idx][["movieId", "title", "genres"]].copy()
    out["similarity"] = sims[top_idx]
    return out


def recommend_for_user(df_ratings, df_movies, user_id, topn=10):
    # Load artifacts
    U = np.load(ART_DIR / "svd_U.npy")
    S = np.load(ART_DIR / "svd_S.npy")
    Vt = np.load(ART_DIR / "svd_Vt.npy")
    user_means = np.load(ART_DIR / "user_means.npy")
    with open(ART_DIR / "user_index.json") as f:
        user_to_idx = json.load(f)
    with open(ART_DIR / "movie_index.json") as f:
        movie_to_idx = json.load(f)

    if str(user_id) not in map(str, user_to_idx.keys()):
        raise ValueError(f"user_id {user_id} not found in dataset.")

    # Align types (json keys are strings)
    user_to_idx = {int(k): int(v) for k, v in user_to_idx.items()}
    movie_to_idx = {int(k): int(v) for k, v in movie_to_idx.items()}

    uidx = user_to_idx[int(user_id)]
    # Pred matrix row for user u: Ŕ_u ≈ U[u] @ diag(S) @ Vt + mean_u
    preds = (U[uidx] * S) @ Vt + float(user_means[uidx])

    # Exclude movies the user already rated
    rated = (
        df_ratings[df_ratings["userId"] == int(user_id)]["movieId"]
        .map(movie_to_idx)
        .dropna()
        .astype(int)
        .tolist()
    )
    preds[rated] = -np.inf

    top_cols = np.argsort(-preds)[:topn]
    inv_movie_idx = {v: k for k, v in movie_to_idx.items()}
    mids = [inv_movie_idx[i] for i in top_cols]
    out = (
        df_movies.set_index("movieId")
        .loc[mids][["title", "genres"]]
        .reset_index()
    )
    out["pred_score"] = preds[top_cols]
    return out


def main():
    parser = argparse.ArgumentParser(description="Movie Recommender (Content + Collaborative)")
    parser.add_argument("--build", action="store_true", help="Download data and build artifacts")
    parser.add_argument("--user", type=int, help="Recommend for a given userId (collaborative)")
    parser.add_argument("--similar", type=str, help="Find movies similar to a given title (content-based)")
    parser.add_argument("--search", type=str, help="Search movie titles")
    parser.add_argument("--topn", type=int, default=10, help="Top-N results")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    ART_DIR.mkdir(exist_ok=True)

    if args.build:
        download_movielens()
        df_ratings, df_movies = load_data()
        build_content_based(df_movies)
        build_collaborative(df_ratings, df_movies)
        print("[i] Build finished.")
        return

    if not (RATINGS_CSV.exists() and MOVIES_CSV.exists()):
        print("[i] Data not found; downloading now…")
        download_movielens()

    df_ratings, df_movies = load_data()
    ensure_built()

    if args.search:
        hits = search_title(df_movies, args.search, args.topn)
        print(hits.to_string(index=False))
        return

    if args.similar:
        res = recommend_similar(df_movies, args.similar, args.topn)
        print(res.to_string(index=False))
        return

    if args.user is not None:
        res = recommend_for_user(df_ratings, df_movies, args.user, args.topn)
        print(res.to_string(index=False))
        return

    print("Nothing to do. Try one of:")
    print("  python recommender.py --build")
    print('  python recommender.py --user 1 --topn 10')
    print('  python recommender.py --similar "Toy Story (1995)" --topn 10')
    print('  python recommender.py --search "toy"')


if __name__ == "__main__":
    main()
