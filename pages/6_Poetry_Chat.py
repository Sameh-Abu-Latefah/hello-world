"""Poetry Chat page: ask questions about the Poetry docs and see the sources."""
import os
import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
# ROOT: for task9.code.rag and src.hello_world.llm. task8/code: for chunking, documents, store.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "task8" / "code"))

from task8.code.chunking import chunk_documents
from task8.code.documents import DOCS_FOLDER, load_documents
from task8.code.store import CHROMA_PATH, build_collection, collection_exists
from task9.code.rag import EMBEDDING_MODEL_NAME, answer_question

load_dotenv()  # local runs: reads GROQ_API_KEY from .env

CHUNK_SIZE = 800
OVERLAP = 150
COLLECTION_NAME = f"poetry_{CHUNK_SIZE}_{OVERLAP}"

st.set_page_config(page_title="Poetry Chat", layout="wide")
st.title("Poetry Chat")
st.caption(
    "Ask a question about Poetry 2.1.4. Each question is answered on its own, "
    "without memory of the previous ones. The sources are shown under each answer."
)


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Preparing the docs. First run only, this can take a minute...")
def ensure_collection() -> str:
    if collection_exists(COLLECTION_NAME, CHROMA_PATH):
        return COLLECTION_NAME
    model = load_model()
    documents = load_documents(DOCS_FOLDER)
    chunks = chunk_documents(documents, CHUNK_SIZE, OVERLAP)
    build_collection(COLLECTION_NAME, chunks, model, path=CHROMA_PATH)
    return COLLECTION_NAME


def get_api_key():
    """Online: st.secrets. Locally: the environment (.env is loaded above)."""
    try:
        key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):  # no secrets file locally, or no such key
        key = None
    if key:
        os.environ["GROQ_API_KEY"] = key  # ask_llm reads it from the environment
        return key
    return os.getenv("GROQ_API_KEY")


_CODE_PART = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)


def clean_answer(text: str) -> str:
    """Make the answer safe and consistent for display.

    - 【3】 becomes [3], so it matches the numbered source expanders.
    - "$" outside code is escaped, so two "$" in a sentence are not drawn as a formula.
    """
    text = re.sub(r"【(\d+)】", r"[\1]", text)
    parts = _CODE_PART.split(text)
    for i in range(0, len(parts), 2):  # even parts are outside code
        parts[i] = parts[i].replace("$", r"\$")
    return "".join(parts)


def show_sources(sources):
    st.caption("Sources")
    for number, source in enumerate(sources, start=1):
        header = f"[{number}] {source['title']}  |  {source['source']}  |  similarity {source['similarity']:.3f}"
        with st.expander(header):
            st.markdown(f"**Title:** {source['title']}")
            st.markdown(f"**File:** `{source['source']}`")
            st.markdown(f"**Similarity:** {source['similarity']:.3f}")
            st.text(source["text"])


def show_message(message):
    if message.get("error"):
        st.error(message["content"])
        return
    st.markdown(message["content"])
    if message.get("sources"):
        show_sources(message["sources"])


# --- Sidebar ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    k = st.slider("Number of sources (k)", min_value=1, max_value=8, value=3)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --- Setup ---
api_key = get_api_key()
if not api_key:
    st.error(
        "The GROQ_API_KEY is missing, so the chat is turned off. "
        "Locally: add it to your .env file. "
        "On Streamlit Cloud: add it in the app's Secrets."
    )

try:
    model = load_model()
    collection_name = ensure_collection()
except Exception as exc:  # noqa: BLE001
    st.error("Could not prepare the docs search. Please reload the page in a moment.")
    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

# --- Chat ---
if not st.session_state.messages:
    st.info("Ask a question about Poetry, for example: How do I add a dev dependency?")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        show_message(message)

question = st.chat_input("Ask a question about Poetry...", disabled=not api_key)

if question and question.strip():
    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the docs..."):
                result = answer_question(question, collection_name, k=k, embed_model=model)
        except Exception as exc: # noqa: BLE001
            reply = {
                "role": "assistant",
                "content": "Sorry, I could not get an answer right now. Please try again in a moment.",
                "error": True,
            }
            show_message(reply)
            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {str(exc)[:300]}")
        else:
            reply = {
                "role": "assistant",
                "content": clean_answer(result["answer"]),
                "sources": result["sources"],
            }
            show_message(reply)
    st.session_state.messages.append(reply)