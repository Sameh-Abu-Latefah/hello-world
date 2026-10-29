import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import APIError
from sklearn.metrics import confusion_matrix

from src.hello_world.llm import ask_llm, parse_label
from src.hello_world.prompts import PROMPT_A, PROMPT_B, PROMPT_C

load_dotenv()

PROMPTS = {"A": PROMPT_A, "B": PROMPT_B, "C": PROMPT_C}
LABELS = ["positive", "negative", "neutral", "invalid"]



def ask_with_retry(system_prompt, comment, model, tries=3):
    for attempt in range(1, tries + 1):
        try:
            return ask_llm(system_prompt, comment, model, temperature=0)
        except APIError as error:
            print(f"  attempt {attempt}/{tries} failed: {error}")
            time.sleep(1)
    return None  # parse_label(None) returns "invalid"


def run_prompt(name, system_prompt, df, model):
    start = time.time()
    predictions = []
    for comment in df["comment"]:
        answer = ask_with_retry(system_prompt, comment, model)
        predictions.append(parse_label(answer))
        time.sleep(0.5)

    out = df[["id", "comment", "label"]].copy()
    out["prediction"] = predictions

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    file_name = results_dir / f"results_{name}_{model.replace('/', '-')}.csv"
    out.to_csv(file_name, index=False)
    print(f"saved {file_name} ({time.time() - start:.1f} seconds)")
    return out


def print_metrics(out):
    accuracy = (out["prediction"] == out["label"]).mean()
    baseline = out["label"].value_counts(normalize=True).max()
    invalid = (out["prediction"] == "invalid").sum()

    print(f"accuracy: {accuracy:.1%}")
    print(f"baseline: {baseline:.1%}")
    print(f"invalid:  {invalid}")

    matrix = confusion_matrix(out["label"], out["prediction"], labels=LABELS)
    print("confusion matrix (rows = correct label, columns = prediction):")
    print(pd.DataFrame(matrix, index=LABELS, columns=LABELS))

    wrong = out[out["prediction"] != out["label"]]
    print("wrong predictions:")
    print(wrong.to_string(index=False))


if __name__ == "__main__":
    model = sys.argv[1]
    names = sys.argv[2:] or list(PROMPTS)

    df = pd.read_csv("../../data/test_comments.csv")

    for name in names:
        print(f"\n=== Prompt {name} | {model} ===")
        results = run_prompt(name, PROMPTS[name], df, model)
        print_metrics(results)