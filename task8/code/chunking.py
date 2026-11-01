def chunk_text(text, chunk_size, overlap):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )
    if not text:
        return []

    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def chunk_documents(documents, chunk_size, overlap):
    chunks = []
    for document in documents:
        pieces = chunk_text(document["text"], chunk_size, overlap)
        for chunk_index, piece in enumerate(pieces):
            chunks.append(
                {
                    "id": f"{document['source']}::{chunk_index}",
                    "text": piece,
                    "source": document["source"],
                    "title": document["title"],
                    "chunk_index": chunk_index,
                }
            )
    return chunks


def main():
    from documents import DOCS_FOLDER, load_documents

    documents = load_documents(DOCS_FOLDER)
    chunks = chunk_documents(documents, chunk_size=300, overlap=0)
    print(f"{len(chunks)} chunks (size 300, overlap 0)")

    first_document = documents[0]["source"]
    sample = [chunk for chunk in chunks if chunk["source"] == first_document][:5]
    for chunk in sample:
        print("\n" + "=" * 60)
        print(f"{chunk['id']}  ({len(chunk['text'])} characters)")
        print(chunk["text"])


if __name__ == "__main__":
    main()