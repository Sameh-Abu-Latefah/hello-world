import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from similarity import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"
ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "test_comments.csv"
EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent / "results" / "embeddings.npy"


def get_embeddings(comments, model, path=EMBEDDINGS_PATH):
    if path.exists():
        saved = np.load(path)
        if saved.shape[0] == len(comments):
            return saved

    embeddings = model.encode(comments)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    return embeddings


def search(query, comments, embeddings, model, k=5, min_score=-1.0):
    query_vector = model.encode(query)
    scores = [cosine_similarity(query_vector, vector) for vector in embeddings]
    best_first = np.argsort(scores)[::-1][:k]
    return [(comments[i], scores[i]) for i in best_first if scores[i] >= min_score]


def split_words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def keyword_search(query, comments, k=5):
    query_words = set(split_words(query))
    scores = [len(query_words & set(split_words(comment))) for comment in comments]
    best_first = np.argsort([-score for score in scores], kind="stable")[:k]
    return [(comments[i], scores[i]) for i in best_first if scores[i] > 0]


def main():
    from sentence_transformers import SentenceTransformer
    if len(sys.argv) < 2:
        sys.exit('usage: python search.py "your query"')
    query = " ".join(sys.argv[1:])

    df = pd.read_csv(CSV_PATH)
    comments = df["comment"].tolist()
    ids = dict(zip(df["comment"], df["id"]))

    model = SentenceTransformer(MODEL_NAME)
    embeddings = get_embeddings(comments, model)

    print(f'Query: "{query}"')
    for rank, (comment, score) in enumerate(search(query, comments, embeddings, model), start=1):
        print(f"{rank}. id={ids[comment]}  score={score:.3f}  {comment}")


if __name__ == "__main__":
    main()