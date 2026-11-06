"""Retrieve chunks from ChromaDB, ask the LLM, and return the answer with its sources."""
import sys
from functools import lru_cache
from pathlib import Path

# store.py lives in task8/code, so add that folder to the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "task8" / "code"))

from src.hello_world.llm import ask_llm
from task8.code.store import query_collection

SYSTEM_PROMPT = (
    "You answer questions about Poetry, the Python dependency manager. "
    "Use the numbered sources below to answer. "
    "After each fact, add the number of the source in square brackets, like [1]. "
    "Keep the answer short, and include the command when there is one."
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Use the same Groq model name you used in Task 6.
LLM_MODEL = "openai/gpt-oss-120b"


@lru_cache(maxsize=1)
def _load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def format_sources(results):
    """Return one text with each source numbered [1], [2], ... plus title, file, text."""
    blocks = []
    for number, result in enumerate(results, start=1):
        metadata = result["metadata"]
        blocks.append(
            f"[{number}] Title: {metadata['title']}\n"
            f"File: {metadata['source']}\n"
            f"Text: {result['text']}"
        )
    return "\n\n".join(blocks)


def build_messages(question, results):
    """Return (system_prompt, user_message). The user message has the sources, then the question."""
    user_message = f"Sources:\n{format_sources(results)}\n\nQuestion: {question}"
    return SYSTEM_PROMPT, user_message


def answer_question(
    question, collection_name, k=3, embed_model=None, llm_model=LLM_MODEL
):
    """Retrieve the top k chunks, ask the LLM with temperature 0, return a dictionary."""
    if embed_model is None:
        embed_model = _load_embedding_model()

    results = query_collection(collection_name, question, embed_model, k=k)
    system_prompt, user_message = build_messages(question, results)
    answer = ask_llm(system_prompt, user_message, llm_model, temperature=0)

    sources = [
        {
            "title": result["metadata"]["title"],
            "source": result["metadata"]["source"],
            "similarity": result["similarity"],
            "text": result["text"],
        }
        for result in results
    ]
    prompt = f"System:\n{system_prompt}\n\nUser:\n{user_message}"

    return {"answer": answer, "sources": sources, "prompt": prompt}