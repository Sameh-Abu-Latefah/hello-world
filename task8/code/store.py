from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = ROOT / "chroma_db"
BATCH_SIZE = 100


def collection_exists(name, path=CHROMA_PATH):
    client = chromadb.PersistentClient(path=str(path))
    return name in [collection.name for collection in client.list_collections()]


def build_collection(name, chunks, model, path=CHROMA_PATH):
    client = chromadb.PersistentClient(path=str(path))

    if collection_exists(name, path):
        client.delete_collection(name)

    collection = client.create_collection(
        name, configuration={"hnsw": {"space": "cosine"}}
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [chunk["text"] for chunk in batch]
        collection.add(
            ids=[chunk["id"] for chunk in batch],
            embeddings=model.encode(texts).tolist(),
            documents=texts,
            metadatas=[
                {
                    "source": chunk["source"],
                    "title": chunk["title"],
                    "chunk_index": chunk["chunk_index"],
                }
                for chunk in batch
            ],
        )

    return collection


def query_collection(name, query, model, k=5, path=CHROMA_PATH):
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_collection(name)

    answer = collection.query(
        query_embeddings=[model.encode(query).tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for text, metadata, distance in zip(
        answer["documents"][0], answer["metadatas"][0], answer["distances"][0]
    ):
        results.append(
            {"text": text, "metadata": metadata, "similarity": 1 - distance}
        )

    results.sort(key=lambda result: result["similarity"], reverse=True)
    return results