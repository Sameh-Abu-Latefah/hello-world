"""Tests for Task 8. They use no real model and no real docs folder."""
import sys
from itertools import pairwise
from pathlib import Path
from typing import ClassVar

import chromadb
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "task8" / "code"))

from task8.code.chunking import chunk_documents, chunk_text
from task8.code.documents import load_documents
from task8.code.run_experiment import hit_position, hit_rate
from task8.code.store import build_collection, collection_exists, query_collection

# ---------------------------------------------------------------- chunk_text

TEXT = "abcdefghij" * 100  # 1000 chars, no spaces, so nothing can be stripped


def test_no_chunk_is_longer_than_chunk_size():
    chunks = chunk_text(TEXT, 100, 20)
    assert chunks
    assert all(len(c) <= 100 for c in chunks)


def test_overlap_zero_joined_chunks_give_original_text():
    chunks = chunk_text(TEXT, 100, 0)
    assert "".join(chunks) == TEXT


def test_overlap_end_of_chunk_is_start_of_next():
    overlap = 20
    chunks = chunk_text(TEXT, 100, overlap)
    assert len(chunks) > 1
    for current, following in pairwise(chunks):
        assert current[-overlap:] == following[:overlap]


def test_empty_text_gives_empty_list():
    assert chunk_text("", 100, 20) == []


def test_overlap_bigger_than_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text(TEXT, 100, 150)


def test_overlap_equal_to_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text(TEXT, 100, 100)


def test_chunk_documents_adds_ids_and_metadata():
    docs = [{"source": "a.md", "title": "A", "text": TEXT}]
    chunks = chunk_documents(docs, 100, 0)
    assert chunks[0]["id"] == "a.md::0"
    assert chunks[1]["id"] == "a.md::1"
    assert chunks[0]["source"] == "a.md"
    assert chunks[0]["title"] == "A"
    assert chunks[1]["chunk_index"] == 1


# ------------------------------------------------------------ load_documents


def test_load_documents_with_and_without_header(tmp_path):
    (tmp_path / "with_header.md").write_text(
        '---\ntitle: "My Title"\nweight: 5\n---\n\nBody of the first file.\n',
        encoding="utf-8",
    )
    (tmp_path / "no_header.md").write_text("Body of the second file.\n", encoding="utf-8")

    docs = {d["source"]: d for d in load_documents(tmp_path)}

    assert set(docs) == {"with_header.md", "no_header.md"}

    with_header = docs["with_header.md"]
    assert with_header["title"] == "My Title"
    assert "Body of the first file." in with_header["text"]
    assert "title:" not in with_header["text"]
    assert "weight" not in with_header["text"]

    no_header = docs["no_header.md"]
    assert "Body of the second file." in no_header["text"]
    assert isinstance(no_header["title"], str)


# ------------------------------------------------------ hit_position / hit_rate


def _result(source, text):
    return {"text": text, "metadata": {"source": source}, "similarity": 0.5}


def test_hit_position_at_1():
    results = [_result("a.md", "has the key phrase")] + [_result("b.md", "x")] * 4
    assert hit_position(results, "a.md", "key phrase") == 1


def test_hit_position_at_4():
    results = [_result("b.md", "x")] * 3 + [_result("a.md", "has the KEY PHRASE")]
    assert hit_position(results, "a.md", "key phrase") == 4


def test_hit_position_not_present():
    results = [
        _result("b.md", "key phrase"),  # right text, wrong file
        _result("a.md", "other text"),  # right file, wrong text
    ]
    assert hit_position(results, "a.md", "key phrase") is None


def test_hit_rate_counts_positions_up_to_k():
    positions = [1, 4, None, 2]
    assert hit_rate(positions, 1) == pytest.approx(1 / 4)
    assert hit_rate(positions, 3) == pytest.approx(2 / 4)
    assert hit_rate(positions, 5) == pytest.approx(3 / 4)


# ---------------------------------------------------------- ChromaDB, no model


class FakeModel:
    """Turns a text into a small hand-made vector. Same shape rules as encode()."""

    VECTORS: ClassVar[dict[str, list[float]]] = {
        "apple": [1.0, 0.0, 0.0],
        "banana": [0.0, 1.0, 0.0],
        "cherry": [0.0, 0.0, 1.0],
    }

    def _vector(self, text):
        for word, vector in self.VECTORS.items():
            if word in text.lower():
                return vector
        return [0.3, 0.3, 0.3]

    def encode(self, texts):
        if isinstance(texts, str):
            return np.array(self._vector(texts))
        return np.array([self._vector(t) for t in texts])


def _small_chunks():
    return [
        {"id": f"{word}.md::0", "text": f"All about {word}.", "source": f"{word}.md",
         "title": word.title(), "chunk_index": 0}
        for word in ("apple", "banana", "cherry")
    ]


def test_build_and_query_with_fake_model(tmp_path):
    model = FakeModel()
    assert not collection_exists("small", tmp_path)

    build_collection("small", _small_chunks(), model, path=tmp_path)
    assert collection_exists("small", tmp_path)

    results = query_collection("small", "apple", model, k=3, path=tmp_path)

    assert len(results) == 3
    assert results[0]["metadata"]["source"] == "apple.md"
    assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-3)
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


def test_query_respects_k(tmp_path):
    model = FakeModel()
    build_collection("small", _small_chunks(), model, path=tmp_path)
    assert len(query_collection("small", "banana", model, k=1, path=tmp_path)) == 1


def test_building_twice_does_not_duplicate(tmp_path):
    model = FakeModel()
    build_collection("small", _small_chunks(), model, path=tmp_path)
    build_collection("small", _small_chunks(), model, path=tmp_path)

    client = chromadb.PersistentClient(path=str(tmp_path))
    assert client.get_collection("small").count() == 3