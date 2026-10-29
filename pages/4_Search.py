import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "task7" / "code"))

from task7.code.search import CSV_PATH, MODEL_NAME, get_embeddings, search

st.title("Search")
st.write("Type a query. The search compares meaning, not just words.")


@st.cache_resource
def load_resources():
    df = pd.read_csv(CSV_PATH)
    comments = df["comment"].tolist()
    model = SentenceTransformer(MODEL_NAME)
    embeddings = get_embeddings(comments, model)
    return df, comments, embeddings, model


df, comments, embeddings, model = load_resources()
info = {row.comment: (row.id, row.label) for row in df.itertuples()}

query = st.text_input("Query", placeholder="for example: money back problem")
k = st.slider("Number of results (k)", min_value=1, max_value=10, value=5)
min_score = st.slider(
    "Minimum similarity score", min_value=0.0, max_value=1.0, value=0.1, step=0.05
)

if not query.strip():
    st.info("Type a query to start.")
else:
    results = search(query, comments, embeddings, model, k=k, min_score=min_score)

    if not results:
        st.warning("No good match found.")
    else:
        table = pd.DataFrame(
            {
                "id": [info[comment][0] for comment, _ in results],
                "comment": [comment for comment, _ in results],
                "label": [info[comment][1] for comment, _ in results],
                "score": [round(float(score), 3) for _, score in results],
            }
        )
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption("The label is the correct label from the dataset, not a prediction.")