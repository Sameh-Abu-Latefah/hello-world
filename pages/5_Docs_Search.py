"""Docs Search page: search the Poetry docs and show where each result comes from."""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "task8" / "code"))

from chunking import chunk_documents
from documents import DOCS_FOLDER, load_documents
from store import CHROMA_PATH, build_collection, collection_exists, query_collection

MODEL_NAME = "all-MiniLM-L6-v2"

# (chunk_size, overlap) settings from Step 7
SETTINGS = [(300, 0), (300, 60), (800, 0), (800, 150), (1500, 0), (1500, 300)]
DEFAULT_SETTING = (800, 150)

st.set_page_config(page_title="Docs Search", layout="wide")
st.title("Docs Search")
st.caption("Search the Poetry 2.1.4 docs. Every result shows its source file.")


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="Preparing the collection. First run only, this can take a minute...")
def ensure_collection(chunk_size: int, overlap: int) -> str:
    name = f"poetry_{chunk_size}_{overlap}"
    if collection_exists(name, CHROMA_PATH):
        return name
    model = load_model()
    documents = load_documents(DOCS_FOLDER)
    chunks = chunk_documents(documents, chunk_size, overlap)
    build_collection(name, chunks, model, path=CHROMA_PATH)
    return name


def setting_label(setting):
    size, overlap = setting
    return f"size {size}, overlap {overlap}"


# --- Controls ---
question = st.text_input("Your question", placeholder="How do I add a dev dependency?")

col_a, col_b, col_c = st.columns(3)
with col_a:
    k = st.slider("Number of results (k)", min_value=1, max_value=10, value=5)
with col_b:
    min_similarity = st.slider(
        "Minimum similarity", min_value=0.0, max_value=1.0, value=0.35, step=0.01
    )
with col_c:
    setting = st.selectbox(
        "Chunk setting",
        SETTINGS,
        index=SETTINGS.index(DEFAULT_SETTING),
        format_func=setting_label,
    )

# --- Search ---
if not question.strip():
    st.info("Type a question to search the docs.")
    st.stop()

collection_name = ensure_collection(*setting)
model = load_model()
results = query_collection(collection_name, question.strip(), model, k=k, path=CHROMA_PATH)
good = [r for r in results if r["similarity"] >= min_similarity]

if not good:
    best = results[0]["similarity"] if results else 0.0
    st.warning("No good match found")
    st.caption(f"Best similarity was {best:.3f}, below the minimum of {min_similarity:.2f}.")
    st.stop()

st.write(f"{len(good)} result(s) above {min_similarity:.2f}")

for rank, result in enumerate(good, start=1):
    meta = result["metadata"]
    title = meta.get("title", "(no title)")
    source = meta.get("source", "(unknown file)")
    header = f"{rank}. {title}  |  {source}  |  similarity {result['similarity']:.3f}"
    with st.expander(header, expanded=(rank == 1)):
        st.markdown(f"**Title:** {title}")
        st.markdown(f"**Source file:** `{source}`")
        st.markdown(f"**Similarity:** {result['similarity']:.3f}")
        st.markdown(f"**Chunk index:** {meta.get('chunk_index', '?')}")
        st.text(result["text"])