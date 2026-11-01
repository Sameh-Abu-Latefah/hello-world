import csv
from pathlib import Path

from chunking import chunk_documents
from documents import DOCS_FOLDER, load_documents
from store import build_collection, query_collection

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = ROOT / "data" / "questions.csv"
RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"

MODEL_NAME = "all-MiniLM-L6-v2"
SETTINGS = [(300, 0), (300, 60), (800, 0), (800, 150), (1500, 0), (1500, 300)]
HIT_KS = (1, 3, 5)
FAIL_K = 5
SHOWN_RESULTS = 3


def load_questions(path=QUESTIONS_PATH):
    with open(path, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return [row for row in rows if (row["expected_source"] or "").strip()]


def is_correct(result, expected_source, key_phrase):
    same_source = result["metadata"]["source"] == expected_source
    has_phrase = key_phrase.lower() in result["text"].lower()
    return same_source and has_phrase


def hit_position(results, expected_source, key_phrase):
    """Return the 1-based position of the first correct result, or None."""
    for position, result in enumerate(results, start=1):
        if is_correct(result, expected_source, key_phrase):
            return position
    return None


def hit_rate(positions, k):
    hits = sum(1 for position in positions if position is not None and position <= k)
    return hits / len(positions)


def describe(result):
    text = " ".join(result["text"].split())[:150]
    return f"{result['metadata']['source']} ({result['similarity']:.3f}): {text}"


def run_setting(chunk_size, overlap, documents, questions, model):
    chunks = chunk_documents(documents, chunk_size, overlap)
    name = f"poetry_{chunk_size}_{overlap}"
    build_collection(name, chunks, model)

    positions = []
    failures = []
    for question in questions:
        results = query_collection(name, question["question"], model, k=max(HIT_KS))
        position = hit_position(
            results, question["expected_source"], question["key_phrase"]
        )
        positions.append(position)

        if position is None or position > FAIL_K:
            row = {
                "id": question["id"],
                "question": question["question"],
                "expected_source": question["expected_source"],
                "key_phrase": question["key_phrase"],
            }
            for number in range(SHOWN_RESULTS):
                row[f"top{number + 1}"] = (
                    describe(results[number]) if number < len(results) else ""
                )
            failures.append(row)

    summary = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunks": len(chunks),
        "average_length": round(sum(len(c["text"]) for c in chunks) / len(chunks)),
    }
    for k in HIT_KS:
        summary[f"hit_at_{k}"] = round(hit_rate(positions, k), 3)
    return summary, failures


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    from sentence_transformers import SentenceTransformer

    documents = load_documents(DOCS_FOLDER)
    questions = load_questions()
    model = SentenceTransformer(MODEL_NAME)
    print(f"{len(documents)} documents, {len(questions)} questions with an answer")

    summaries = []
    for chunk_size, overlap in SETTINGS:
        summary, failures = run_setting(
            chunk_size, overlap, documents, questions, model
        )
        summaries.append(summary)
        print(
            f"\npoetry_{chunk_size}_{overlap}: {summary['chunks']} chunks, "
            f"average length {summary['average_length']}"
        )
        print(
            "  "
            + ", ".join(f"hit@{k} {summary[f'hit_at_{k}']:.3f}" for k in HIT_KS)
            + f"  ({len(failures)} failed)"
        )
        save_csv(
            RESULTS_FOLDER / f"failures_{chunk_size}_{overlap}.csv",
            failures,
            ["id", "question", "expected_source", "key_phrase", "top1", "top2", "top3"],
        )

    save_csv(RESULTS_FOLDER / "experiments_summary.csv", summaries, list(summaries[0]))
    print(f"\nSaved results in {RESULTS_FOLDER}")


if __name__ == "__main__":
    main()