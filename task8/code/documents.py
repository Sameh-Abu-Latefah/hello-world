import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_FOLDER = ROOT / "data" / "poetry_docs"


def split_header(raw_text):
    """Return (header_lines, body). The header sits between two '---' lines at the top."""
    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return [], raw_text

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return lines[1:index], "\n".join(lines[index + 1 :])

    return [], raw_text


def get_title(header_lines, default):
    for line in header_lines:
        if line.startswith("title:"):
            title = line.split(":", 1)[1].strip().strip("\"'")
            return title or default
    return default


def clean_text(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(folder):
    documents = []
    for path in sorted(Path(folder).glob("*.md")):
        raw_text = path.read_text(encoding="utf-8-sig")
        header_lines, body = split_header(raw_text)
        documents.append(
            {
                "source": path.name,
                "title": get_title(header_lines, default=path.name),
                "text": clean_text(body),
            }
        )
    return documents


def main():
    documents = load_documents(DOCS_FOLDER)
    print(f"Loaded {len(documents)} documents")

    for document in documents[:2]:
        print("\n" + "=" * 60)
        print(f"source: {document['source']}")
        print(f"title:  {document['title']}")
        print(f"length: {len(document['text'])} characters")
        print("first 5 lines of text:")
        for line in document["text"].splitlines()[:5]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
    