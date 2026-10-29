import sys

import pandas as pd
from search import (
    CSV_PATH,
    MODEL_NAME,
    get_embeddings,
    keyword_search,
    search,
)
from sentence_transformers import SentenceTransformer

DEFAULT_QUERIES = [
    "slow delivery",
    "broken product",
    "helpful customer service",
    "money back problem",
    "good value",
    "quantum physics",
    "invoice",
    "late arrival",
]
WIDTH = 58


def format_line(item, ids, score_format):
    comment, score = item
    line = f"id={ids[comment]:<2} {format(score, score_format):>5}  {comment}"
    return line[:WIDTH]


def print_side_by_side(query, semantic, keyword, ids):
    left = [format_line(item, ids, ".3f") for item in semantic]
    right = [format_line(item, ids, "d") for item in keyword]
    if not right:
        right = ["(no comment shares a word with the query)"]

    print(f'\nQuery: "{query}"')
    print(f"{'SEMANTIC':<{WIDTH}} | KEYWORD")
    print("-" * WIDTH + "-+-" + "-" * WIDTH)
    for i in range(max(len(left), len(right))):
        left_text = left[i] if i < len(left) else ""
        right_text = right[i] if i < len(right) else ""
        print(f"{left_text:<{WIDTH}} | {right_text}")


def main():
    queries = sys.argv[1:] or DEFAULT_QUERIES

    df = pd.read_csv(CSV_PATH)
    comments = df["comment"].tolist()
    ids = dict(zip(df["comment"], df["id"]))

    model = SentenceTransformer(MODEL_NAME)
    embeddings = get_embeddings(comments, model)

    for query in queries:
        semantic = search(query, comments, embeddings, model)
        keyword = keyword_search(query, comments)
        print_side_by_side(query, semantic, keyword, ids)


if __name__ == "__main__":
    main()