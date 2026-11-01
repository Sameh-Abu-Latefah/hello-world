import csv
from pathlib import Path

from documents import DOCS_FOLDER, load_documents

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = ROOT / "data" / "questions.csv"
MAX_PHRASE_LENGTH = 60


def main():
    documents = load_documents(DOCS_FOLDER)
    texts = {document["source"]: document["text"].lower() for document in documents}

    with open(QUESTIONS_PATH, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    problems = 0
    seen_ids = set()
    answerable = 0

    for row in rows:
        question_id = row["id"]
        source = (row["expected_source"] or "").strip()
        phrase = (row["key_phrase"] or "").strip().lower()
        label = f"[id {question_id}]"

        if question_id in seen_ids:
            print(f"{label} duplicate id")
            problems += 1
        seen_ids.add(question_id)

        if not row["question"].strip():
            print(f"{label} empty question")
            problems += 1

        if not source and not phrase:
            continue

        answerable += 1

        if source not in texts:
            print(f"{label} unknown file: {source!r}")
            problems += 1
            continue

        if not phrase:
            print(f"{label} has a source but no key_phrase")
            problems += 1
            continue

        if phrase not in texts[source]:
            print(f"{label} key_phrase not found in {source}: {phrase!r}")
            problems += 1

        if len(phrase) > MAX_PHRASE_LENGTH:
            print(f"{label} key_phrase is long ({len(phrase)} chars), a cut may split it")

        other_files = [
            name for name, text in texts.items() if name != source and phrase in text
        ]
        if other_files:
            print(f"{label} key_phrase also appears in: {', '.join(other_files)}")

    print(
        f"\n{len(rows)} questions: {answerable} with an answer, "
        f"{len(rows) - answerable} without. Problems: {problems}"
    )


if __name__ == "__main__":
    main()