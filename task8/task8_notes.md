# Task 8: Documents, Chunking, and a Vector Database

## Step 1: Get the documents

Poetry 2.1.4 docs, from the zip Mohammad shared. 16 `.md` files in `data/poetry_docs/`.

## Step 2: Look at the data before you touch it

- Files: 16
- Total size: 224,878 characters
- Longest: `cli.md` (40,855). Shortest: `community.md` (654).

**Header block:** each of the 16 files starts with a YAML block between two `---` lines.
Most files have `title`, `draft`, `type`, `layout`, and `menu` (with `docs` and `weight` inside it).

Differences that `load_documents` must handle:

- `managing-dependencies.md` has the keys in a different order, and its `title` has no quotes.
- `contributing.md` has an extra `note` key. Its value contains a `:`, so split each line only on the first `:`.
- There are 1 or 2 blank lines after the header. `libraries.md`, `managing-dependencies.md`, and `cli.md` have 2.
**Length:** `cli.md` (40,855), `pyproject.md` (25,048), and `dependency-specification.md` (23,224) are the longest. `community.md` (654) and `libraries.md` (4,093) are the shortest. The top 3 files hold about 40% of all characters.

**Code, tables, and tags:** yes, all three appear.

- About 380 code blocks, mostly in `cli.md`, `pyproject.md`, and `dependency-specification.md`.
- Only 7 tables, in `dependency-specification.md`, `cli.md`, and `configuration.md`.
- Hugo shortcodes (`{{% note %}}`, `{{< tabs >}}`, `{{< relref "..." >}}`) appear more than 600 times.

**Three things that could make search harder:**

1. **Hugo tags inside the text.** The files have more than 600 shortcodes such as `{{% note %}}`, `{{< tabs >}}`, and `{{< relref "pyproject#extras" >}}`. The embedding model reads them as normal text, so they add noise and dilute the meaning of a chunk. `cli.md` alone has 104 of them.
2. **Code blocks.** There are about 380, mostly in `cli.md` (89), `pyproject.md` (63), and `dependency-specification.md` (46). A chunk that is mostly TOML or bash is hard to match with a question in plain English. A fixed-size cut can also split a block in the middle.
3. **Repeated headers and repeated content.** `pyproject.md` has the headers `name`, `version`, `description`, `license`, `authors`, and others twice, once under `project` and once under `tool.poetry`. `dependency-specification.md` shows many examples twice, in a `[project]` tab and a `[tool.poetry]` tab. A question can then have two valid answers, and the search can return the wrong copy.

## Step 6: Test questions

I wrote 12 questions: 10 with an answer and 2 with no answer in the docs.

**Hardest for the search:** ...

## Step 7: Run the chunking experiments

Results from 9 questions that have an answer.

| Chunk size | Overlap | Number of chunks | Hit at 1 | Hit at 3 | Hit at 5 |
| ---------- | ------- | ---------------- | -------- | -------- | -------- |
| 300        | 0       | 752              | 0.222    | 0.222    | 0.222    |
| 300        | 60      | 934              | 0.000    | 0.111    | 0.222    |
| 800        | 0       | 286              | 0.444    | 0.444    | 0.444    |
| 800        | 150     | 348              | 0.556    | 0.556    | 0.667    |
| 1500       | 0       | 156              | 0.222    | 0.556    | 0.667    |
| 1500       | 300     | 190              | 0.444    | 0.556    | 0.667    |


## Step 8: Read the failures

I read the failures file of the best setting, `failures_800_150.csv`. Three questions failed.

### Failures

**Question 1:** "How do I add a package that is only needed while developing?"
The search found the right file, but the wrong chunk.
The question says "developing". The docs say "group".
Group: **C (words)**.

**Question 2:** "My teammates and I work on the same Python project... How do we make everyone get the same setup?"
The search did not return `basic-usage.md` at all.
The question has none of the words used in the docs, like "lock" or "dependencies".
Group: **C (words)**.

**Question 6:** "Every time I change one source file, Docker reinstalls all my third-party packages."
The search found the right topic in `faq.md`. The problem is in one chunk, and the solution is in the next chunk.
Group: **A (split)**.

### Count

| Group | Failures |
| ----- | -------- |
| A: split | 1 |
| B: noisy | 0 |
| C: words | 2 |
| D: bad test | 0 |

### The change I would make

Split the text at headings, and add the file title and the section heading at the start of every chunk.
This helps group C, because the chunk gets more of the words a user would use.
It also helps group A, because a solution stays with its heading.
I did not make this change yet.

### Questions with no answer

(To fill after I check the similarity scores.)

## Step 8 – No-answer questions

| Group | Top-1 similarity |
|---|---|
| Answerable (9) | 0.397 – 0.770 |
| No-answer (2) | 0.521, 0.622 |

**Result:** The ranges overlap. No threshold separates them.

**Why:** Both no-answer questions are about Poetry. They are close in topic to the docs. The model measures topic closeness, not whether an answer exists.

**Cost of a high threshold:** 0.65 blocks both no-answer questions but also drops 6 of 9 good questions.

**Decision:** Default threshold 0.35, adjustable by the user. Scores are shown next to every result.






