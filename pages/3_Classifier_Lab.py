import os
from pathlib import Path

import pandas as pd
import streamlit as st
from groq import APIError
from sklearn.metrics import confusion_matrix

from src.hello_world.llm import ask_llm, parse_label
from src.hello_world.prompts import PROMPT_A, PROMPT_B, PROMPT_C

PROMPTS = {"A": PROMPT_A, "B": PROMPT_B, "C": PROMPT_C}
LABELS = ["positive", "negative", "neutral", "invalid"]
MODEL = "openai/gpt-oss-120b"
ROOT = Path(__file__).resolve().parent.parent


def load_api_key():
    try:
        key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        key = None
    return key or os.getenv("GROQ_API_KEY")


st.title("Classifier Lab")

st.header("Try it")
comment = st.text_area("Comment", placeholder="Type one customer comment")
prompt_name = st.selectbox("Prompt", list(PROMPTS))

if st.button("Classify"):
    key = load_api_key()
    if not comment.strip():
        st.warning("Type a comment first.")
    elif not key:
        st.error("The API key is missing. Add GROQ_API_KEY to the app secrets.")
    else:
        os.environ["GROQ_API_KEY"] = key
        try:
            with st.spinner("Asking the model..."):
                answer = ask_llm(PROMPTS[prompt_name], comment, MODEL)
        except APIError:
            st.error("The model service is not available right now. Try again soon.")
        else:
            st.success(f"Predicted label: {parse_label(answer)}")

st.header("Saved results")
files = sorted((ROOT / "task6" / "results").glob("results_*.csv"))
if not files:
    st.info("No results files found. Run evaluate.py and commit the CSV files.")

for file in files:
    out = pd.read_csv(file)
    st.subheader(file.stem)

    accuracy = (out["prediction"] == out["label"]).mean()
    st.metric("Accuracy", f"{accuracy:.1%}")

    matrix = confusion_matrix(out["label"], out["prediction"], labels=LABELS)
    st.caption("Confusion matrix (bars = correct label, colors = prediction)")
    st.bar_chart(pd.DataFrame(matrix, index=LABELS, columns=LABELS))

    wrong = out[out["prediction"] != out["label"]]
    st.write(f"Wrong predictions: {len(wrong)}")
    st.dataframe(wrong, use_container_width=True)