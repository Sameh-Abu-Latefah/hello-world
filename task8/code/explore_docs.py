from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_FOLDER = ROOT / "data" / "poetry_docs"
PREVIEW_LINES = 15


def read_files(folder):
    rows = []
    for path in sorted(folder.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        rows.append((path.name, len(text), len(text.splitlines())))
    return rows


def main():
    rows = read_files(DOCS_FOLDER)
    if not rows:
        raise SystemExit(f"No .md files found in {DOCS_FOLDER}")

    rows.sort(key=lambda row: row[1], reverse=True)

    print(f"{'file':<35} {'characters':>10} {'lines':>7}")
    print("-" * 54)
    for name, characters, lines in rows:
        print(f"{name:<35} {characters:>10} {lines:>7}")

    total_characters = sum(characters for _, characters, _ in rows)
    print("-" * 54)
    print(f"Total files: {len(rows)}")
    print(f"Total characters: {total_characters}")

    longest_name = rows[0][0]
    print(f"\nFirst {PREVIEW_LINES} lines of {longest_name}:\n")
    longest_text = (DOCS_FOLDER / longest_name).read_text(encoding="utf-8")
    for line in longest_text.splitlines()[:PREVIEW_LINES]:
        print(line)


if __name__ == "__main__":
    main()