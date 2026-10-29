import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "task7" / "code"))

from task7.code.search import keyword_search, search
from task7.code.similarity import cosine_similarity


class FakeModel:
    """Stands in for SentenceTransformer. No download, no real embeddings."""

    def __init__(self, vectors_by_text):
        self.vectors_by_text = vectors_by_text

    def encode(self, text):
        return np.array(self.vectors_by_text[text], dtype=float)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([1, 2, 3], [1, 2, 3], 1.0),
        ([1, 0], [0, 1], 0.0),
        ([1, 0], [-1, 0], -1.0),
    ],
    ids=["identical", "perpendicular", "opposite"],
)
def test_cosine_similarity_known_vectors(a, b, expected):
    assert cosine_similarity(a, b) == pytest.approx(expected)


def test_search_ranks_from_highest_to_lowest():
    comments = ["first", "second", "third"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    model = FakeModel({"query": [1.0, 0.2]})

    results = search("query", comments, embeddings, model, k=3)

    assert [comment for comment, _ in results] == ["first", "third", "second"]
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_search_returns_only_k_results():
    comments = ["first", "second", "third"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    model = FakeModel({"query": [1.0, 0.2]})

    results = search("query", comments, embeddings, model, k=2)

    assert [comment for comment, _ in results] == ["first", "third"]


def test_search_with_k_larger_than_number_of_comments():
    comments = ["first", "second", "third"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    model = FakeModel({"query": [1.0, 0.2]})

    results = search("query", comments, embeddings, model, k=10)

    assert len(results) == 3


def test_keyword_search_with_k_larger_than_number_of_comments():
    comments = ["red apple", "green apple"]

    results = keyword_search("apple", comments, k=10)

    assert len(results) == 2


def test_keyword_search_ignores_capital_letters():
    comments = ["The Package ARRIVED late", "Nothing relevant here"]

    upper_comment = keyword_search("package arrived", comments)
    upper_query = keyword_search("PACKAGE Arrived", ["the package arrived late"])

    assert upper_comment == [("The Package ARRIVED late", 2)]
    assert upper_query == [("the package arrived late", 2)]