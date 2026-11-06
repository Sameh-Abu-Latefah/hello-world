"""Run all questions from data/questions.csv in plain mode or in RAG mode.

The answers are saved to task9/results/answers_MODE_kK.csv. Grade them from the
saved file. Do not call the API again for the same run.

Run from the project root:
    python task9/code/run_rag_eval.py --mode plain
    python task9/code/run_rag_eval.py --mode rag --k 3
"""
import argparse
import csv
import sys
import time
from pathlib import Path

from groq import APIError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import rag

from src.hello_world.llm import ask_llm

QUESTIONS_FILE = ROOT / "data" / "questions.csv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
COLLECTION = "poetry_800_150"
PLAIN_SYSTEM_PROMPT = (
    "You answer questions about Poetry, the Python dependency manager. "
    "Keep the answer short, and include the command when there is one."
)
COLUMNS = [
    "id",
    "question",
    "expected_source",
    "key_phrase",
    "answer",
    "retrieved_sources",
    "retrieval_hit",
]


def read_questions():
    with open(QUESTIONS_FILE, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return [row for row in rows if (row.get("question") or "").strip()]


def with_retry(function, *args, tries=3, **kwargs):
    """Call function. Retry on an API error. Return None if every try fails."""
    for attempt in range(1, tries + 1):
        try:
            return function(*args, **kwargs)
        except APIError as error:
            print(f"  attempt {attempt}/{tries} failed: {error}")
            time.sleep(1)
    return None


def is_hit(sources, expected_source, key_phrase):
    """True if one retrieved chunk has the expected source and contains the key phrase."""
    return any(
        source["source"] == expected_source
        and key_phrase.lower() in source["text"].lower()
        for source in sources
    )


def run_question(mode, k, question):
    """Return (answer, sources). The answer is an empty string if the API failed."""
    if mode == "plain":
        answer = with_retry(
            ask_llm, PLAIN_SYSTEM_PROMPT, question, rag.LLM_MODEL, temperature=0
        )
        return answer or "", []

    result = with_retry(rag.answer_question, question, COLLECTION, k=k)
    if result is None:
        return "", []
    return result["answer"] or "", result["sources"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["plain", "rag"], required=True)
    parser.add_argument("--k", type=int, default=3, help="chunks to retrieve (rag mode)")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing answers file"
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"answers_{args.mode}_k{args.k}.csv"
    if output_file.exists() and not args.overwrite:
        print(f"{output_file} already exists. Work from the saved file.")
        print("Use --overwrite only if you really want to call the API again.")
        return

    questions = read_questions()
    rows = []
    failed = []

    for number, question in enumerate(questions, start=1):
        print(f"[{number}/{len(questions)}] question {question['id']}")
        answer, sources = run_question(args.mode, args.k, question["question"])
        if not answer.strip():
            failed.append(question["id"])

        expected_source = (question.get("expected_source") or "").strip()
        key_phrase = (question.get("key_phrase") or "").strip()
        if args.mode == "rag" and expected_source and sources:
            retrieval_hit = str(is_hit(sources, expected_source, key_phrase))
        else:
            retrieval_hit = ""

        rows.append(
            {
                "id": question["id"],
                "question": question["question"],
                "expected_source": expected_source,
                "key_phrase": key_phrase,
                "answer": answer,
                "retrieved_sources": ";".join(source["source"] for source in sources),
                "retrieval_hit": retrieval_hit,
            }
        )
        time.sleep(0.5)

    with open(output_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nsaved {output_file} ({len(rows)} rows)")
    if failed:
        print(f"WARNING: no answer for question ids: {', '.join(failed)}")


if __name__ == "__main__":
    main()