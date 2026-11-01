"""Print top similarity scores for answerable and no-answer questions.

Goal: see if a minimum-similarity threshold can separate them.
Run from the project root:  python task8/code/no_answer_scores.py
"""
import csv
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))

from store import CHROMA_PATH, collection_exists, query_collection

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "data" / "questions.csv"
COLLECTION = "poetry_800_150"
K = 5


def read_questions():
    with open(QUESTIONS, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def is_hit(result, expected_source, key_phrase):
    same_source = result["metadata"]["source"] == expected_source
    has_phrase = key_phrase.lower() in result["text"].lower()
    return same_source and has_phrase


def main():
    if not collection_exists(COLLECTION, CHROMA_PATH):
        print(f"Collection {COLLECTION} not found. Run run_experiments.py first.")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    rows = read_questions()

    answerable = []  # (id, top1_similarity, correct_similarity or None)
    no_answer = []  # (id, top1_similarity, top1_source)

    for row in rows:
        results = query_collection(COLLECTION, row["question"], model, k=K, path=CHROMA_PATH)
        top1 = results[0]["similarity"]
        expected = (row.get("expected_source") or "").strip()
        phrase = (row.get("key_phrase") or "").strip()

        if expected and phrase:
            correct = None
            for r in results:
                if is_hit(r, expected, phrase):
                    correct = r["similarity"]
                    break
            answerable.append((row["id"], top1, correct))
        else:
            no_answer.append((row["id"], top1, results[0]["metadata"]["source"]))

    print(f"\nCollection: {COLLECTION}\n")
    print("ANSWERABLE questions")
    print(f"{'id':<4} {'top1':>6} {'correct chunk':>14}")
    for qid, top1, correct in answerable:
        shown = f"{correct:.3f}" if correct is not None else "not in top 5"
        print(f"{qid:<4} {top1:>6.3f} {shown:>14}")

    print("\nNO-ANSWER questions")
    print(f"{'id':<4} {'top1':>6}  source of top1")
    for qid, top1, source in no_answer:
        print(f"{qid:<4} {top1:>6.3f}  {source}")

    if not answerable or not no_answer:
        print("\nNeed at least one answerable and one no-answer question.")
        return

    ans_tops = [t for _, t, _ in answerable]
    no_tops = [t for _, t, _ in no_answer]
    print("\nSUMMARY (top-1 similarity)")
    print(f"answerable : min {min(ans_tops):.3f}  max {max(ans_tops):.3f}")
    print(f"no-answer  : min {min(no_tops):.3f}  max {max(no_tops):.3f}")

    if max(no_tops) < min(ans_tops):
        mid = (max(no_tops) + min(ans_tops)) / 2
        print(f"Clean gap. Threshold between them: about {mid:.2f}")
    else:
        print("Overlap: no threshold separates them perfectly.")
        for t in (0.3, 0.35, 0.4, 0.45, 0.5):
            kept = sum(1 for x in ans_tops if x >= t)
            blocked = sum(1 for x in no_tops if x < t)
            print(
                f"  threshold {t:.2f}: keeps {kept}/{len(ans_tops)} answerable, "
                f"blocks {blocked}/{len(no_tops)} no-answer"
            )


if __name__ == "__main__":
    main()