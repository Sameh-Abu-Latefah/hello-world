---
tags: [internship, ai-training, task9, rag]
---

# Task 9 – My First RAG Chatbot

> [!info] About this file
> All numbers come from my real runs.
> `==[Fill in ...]==` means I must add the real result.
> My `questions.csv` has **11 questions, not 30**. All numbers are for these 11 (9 with an answer in the docs, 2 without).

---

## Introduction

**The idea.** I built a chatbot that answers questions about Poetry. It first searches the Poetry 2.1.4 docs. Then it answers and shows its sources.

**What is RAG?** RAG means *Retrieval Augmented Generation*.
- **Retrieval:** I search the docs and take the most similar pieces (chunks).
- **Generation:** An AI model reads those chunks and writes the answer.

**Why combine them?** A model alone answers from memory. Its memory can be old or wrong. With the real docs in front of it, the model can give better answers. The user can also check the sources.

**The goal.** Build the chatbot, test it, grade the answers, and find out why some fail.

**The final project has:**
- `task9/code/rag.py` – the RAG function
- `task9/code/run_rag_eval.py` – runs all questions
- `task9/results/` – saved answers
- `pages/6_Poetry_Chat.py` – the chat page
- `tests/test_rag.py` – the tests

---

## Step 1 – Preparation

| Setting | Value | Why |
|---|---|---|
| Chunk setting | 800 size, 150 overlap (`poetry_800_150`) | The setting from Task 8. ==[Fill in: why Task 8 chose it]== |
| Groq model | `openai/gpt-oss-120b` | The model from my Task 6 results. |
| Temperature | 0 | The answer is the most stable. The same question gives almost the same answer. |

**What I checked:**
- `GROQ_API_KEY` is in `.env`. The code reads it inside `ask_llm`.
- `.env` is not tracked by git. `git ls-files .env` gave nothing.
- The key is not in the git history. `git log -S "gsk_" --all --oneline` gave nothing.
- `questions.csv` should have 30 questions. ==[Fill in: it has 11 now]==
- `ask_llm` (Task 6) and `query_collection` (Task 8) both worked in my first RAG run.

**Why the same settings?** I compare plain mode and RAG mode. If I change the model, temperature or chunks, I will not know what caused the difference. So I change only one thing: the docs.

**Observation.** The only problem was a wrong model name (`llama-3.3-70b-versatile`). Groq returned `404 model_not_found`. I fixed it with the name from Task 6.

---

## Step 2 – The RAG Function

`rag.py` connects the search (Task 8) and the model (Task 6). I give it a question. It returns an answer and its sources.

### The five steps

1. I ask a question.
2. The question becomes an embedding (`all-MiniLM-L6-v2`).
3. ChromaDB returns the top `k` similar chunks.
4. The chunks are written as numbered sources `[1]`, `[2]`, `[3]` and put in the message with the question.
5. The Groq model answers and writes source numbers in the answer.

```mermaid
flowchart LR
    Q[Question] --> E[Embedding]
    E --> S[Top k chunks]
    S --> F[Numbered sources]
    F --> M[Prompt]
    M --> L[Groq model]
    L --> A[Answer with citations]
```

### The three functions

**`format_sources(results)`** turns the chunks into one text. Numbers start at 1. Each source has its title, file name, and text:

```text
[1] Title: Managing dependencies
File: managing-dependencies.md
Text: ...chunk text...
```

**`build_messages(question, results)`** returns two texts.
- The *system prompt* is fixed. It tells the model to use the numbered sources, add `[1]` after each fact, keep the answer short, and give the command.
- The *user message* has `Sources:` first, then the sources, then `Question:` at the end.

**`answer_question(question, collection_name, k=3)`** runs everything. It searches, builds the messages, calls the model with temperature 0, and returns a dictionary:

| Key | Content |
|---|---|
| `answer` | The model's text |
| `sources` | A list. Each item has `title`, `source` (file), `similarity`, `text` |
| `prompt` | The full text I sent. It helps me debug |

**Citations.** `[1]` tells me which source the model used. So I can open it and check. But it is only a claim by the model. It is not proof.

### Test of three questions

| Question | What I saw | Do the sources support the answer? |
|---|---|---|
| How do I add a dev dependency? | The first answer was wrong. It used `--dev-dependency`. That option is from `poetry init`, not `poetry add`. | No. A chunk about `poetry init` options was retrieved and used in the wrong place. |
| ==[Fill in: question 2]== | ==[Fill in]== | ==[Fill in]== |
| ==[Fill in: question 3]== | ==[Fill in]== | ==[Fill in]== |

**I learned:** good retrieval can still give a wrong answer if the model uses a chunk in the wrong place.

---

## Step 3 – Two Modes

| | Plain mode | RAG mode |
|---|---|---|
| Model gets | Only the question | The question + `k` numbered sources |
| Knowledge from | Its memory | The Poetry docs |
| Citations | No | Yes |

The model and temperature are the same in both modes.

### `run_rag_eval.py`

It reads all questions from `data/questions.csv`, asks each one, and saves the answers.
- `--mode plain` or `--mode rag` chooses the mode.
- `--k 3` sets the number of chunks (default 3).
- `--overwrite` is needed to replace an old file. This protects my results.
- It waits **0.5 seconds** between questions.
- If Groq gives an API error, it **retries up to 3 times**. If all fail, the answer stays empty and a warning is printed.
- Output: `task9/results/answers_plain_k3.csv` and `answers_rag_k3.csv`. Each has 11 rows now. ==[Fill in: 30 rows later]==

### CSV columns

| Column | Meaning |
|---|---|
| `id` | Question number |
| `question` | The question |
| `expected_source` | The doc file with the answer (empty if no answer exists) |
| `key_phrase` | A phrase from the docs that should be in a retrieved chunk |
| `answer` | The model's answer |
| `retrieved_sources` | Files of the retrieved chunks, separated by `;` |
| `retrieval_hit` | `True` if a retrieved chunk is from the expected file **and** has the key phrase |

**Why save the answers?** A new API call costs time and can give a different answer. A saved file never changes. So I grade and compare the same answers.

---

## Step 4 – Manual Grading

**Why by hand?** A good-sounding answer can still be wrong. I must compare it with the real docs.

| Grade | Meaning |
|---|---|
| **Right** | Correct. Nothing important is missing. |
| **Partly** | The main idea is right, but a detail is wrong or missing. |
| **Wrong** | Incorrect, made up, or does not answer. |

**`citation_ok`** checks if each citation is really supported by the cited file.
- `yes` – all citations are supported.
- `no` – at least one is not supported.
- `n/a` – there are no citations (always so in plain mode).

### Results

| Mode | Right | Partly | Wrong | Citations Correct |
|---|---:|---:|---:|---:|
| Plain Model | 6 | 5 | 0 | n/a |
| RAG (k=3) | 8 | 2 | 1 | 9 of 11 |

**Did RAG do better?** A little. It had 2 more right answers (8 vs 6).

**Where plain was better:**
- Q6 (Docker): RAG forgot the last step that installs the project.
- Q11 (AWS Lambda, no answer in docs): plain was "partly". RAG was "wrong".

**Did plain give old commands?** Yes, but not invented ones.
- Q1: it used the old `[tool.poetry.dev-dependencies]`.
- Q4: it used `poetry install --sync`. It is deprecated. The new command is `poetry sync`.
- Q11: it used `poetry export`. Since Poetry 2.0 it needs a plugin.

**How reliable is this?** Not very. One question is about 9 points. Two plain grades (Q7, Q8) are close calls. If they were "right", both modes would have 8. ==[Fill in after 30 questions]==

### Three examples

**RAG helped – Q4 (match the lock file).** Plain said `poetry install --sync` (deprecated). RAG said `poetry sync` and cited `managing-dependencies.md`.

**No big difference – Q3 (bug report).** Both answers were right. ==[Fill in: one sentence on how they looked different]==

**RAG was worse – Q11 (AWS Lambda).** The docs do not cover Lambda. RAG made up zip steps and its citation `[1]` did not support them. Q6 was also less complete with RAG.

---

## Step 5 – Why Answers Fail

| Type | Meaning |
|---|---|
| **Retrieval failure** | The right chunk was not retrieved. |
| **Generation failure** | The right chunk was there, but the answer is still wrong. |
| **Bad test** | My question or expected source was wrong. |

**How `retrieval_hit` helps.** `False` means the right chunk never reached the model. `True` with a bad answer means a generation problem.

### My results (RAG k=3, "partly" or "wrong")

| Failure Type | Number of Cases | Explanation |
|---|---:|---|
| Retrieval Failure | 2 | Q1 and Q6. `retrieval_hit` was `False`. In Q6 the answer had no final step that installs the project. ==[Fill in: what was missing in Q1]== |
| Generation Failure | 1 | Q11. The docs have no answer. The model made one up. The prompt does not say what to do when sources have no answer. |
| Bad Test | 0 | My questions and expected sources were correct. |

### k = 1, 3, 5

| | k=1 | k=3 | k=5 |
|---|---:|---:|---:|
| Right | 9 | 8 | 7 |
| Partly | 2 | 2 | 3 |
| Wrong | 0 | 1 | 1 |
| Retrieval hit (of 9) | 5 | 5 | 6 |

- **Q11 (no answer):** k=1 refused correctly. With k=3 and k=5, the Docker FAQ chunk came in and the model built a Lambda guide.
- **Q9:** at k=5 the answer told end users to use `poetry install -E`. This was a mistake.
- **Q1 and Q6:** stayed "partly" at every k. The right chunk was never retrieved.

**Did more chunks always help?** No. Right answers went down: 9, 8, 7. Extra context gave the model material to make mistakes.

**Retrieval or prompt?** Retrieval needs more work. When the right chunk was retrieved, the answer was right 5 times out of 5. When it was missed, only 2 out of 4. The prompt is the second problem: it never says "if the sources have no answer, say so". But my sample is tiny (11 questions).

---

## Step 6 – Special Test Questions

**Why important?** Normal questions show that the chatbot can answer. Special questions show what it does when it should *not* answer. Real users ask these too.

I used `answer_question` with k=3. I did not change the code or the prompt. I only watched.

**The five questions:**
- **A.** What is the capital of France?
- **B.** What does the command poetry deploy-to-cloud do?
- **C.** Why does Poetry run my unit tests in parallel by default?
- **D.** How do I deploy my project to AWS Lambda using only a Poetry command?
- **E.** Which is better, Poetry or another Python packaging tool, and why?

**Two no-answer questions from Task 8:**
- Task 8 Q1: Can Poetry generate API documentation?
- Task 8 Q2: How do I deploy my Poetry project to AWS Lambda?

| Question | Is the Answer in the Docs? | Chatbot Answer | Cited Sources | Do the Sources Support the Answer? |
|---|---|---|---|---|
| A | No | "Paris." | None | n/a. The 3 chunks were unrelated (similarity 0.138). It used its own knowledge. |
| B | No | Refused. No source has this command. | None | n/a. The refusal is true. |
| C | No (wrong idea in the question) | Poetry does not run tests in parallel. If they do, `pytest-xdist` may be the reason. | [1] `contributing.md` | No. The file only shows `poetry run pytest`. `pytest-xdist` is the model's own knowledge. |
| D | No | Refused. The sources do not have this. | None | n/a. The refusal is true. |
| E | No | Poetry is "more streamlined" and "often preferred". | [1], [2] `_index.md` | Partly. The file supports the facts. It does not say Poetry is better. That is an opinion. |
| Task 8 – Question 1 | No | Poetry does not generate API docs. Use Sphinx or MkDocs. | [2] `_index.md` | Partly. The file says Poetry is for dependencies and packaging. The tool names are the model's knowledge. |
| Task 8 – Question 2 | No | A 4-step guide: install Poetry, Docker image, zip, upload. | [1] `contributing.md`, [2] `_index.md`, [3] `faq.md` | No. Only the install command and a Docker pattern exist in the docs. |

Best similarity: A 0.138, B 0.568, C 0.532, D 0.545, E 0.699.

**Why check the docs myself?** The chatbot sounds sure even when it is wrong. I searched the docs for `deploy`, `parallel` and `Lambda` to be sure. The only "parallel" is `installer.parallel`, about installing packages, not tests.

### Reflection

The chatbot did not treat all questions the same way. It answered A from memory, refused B and D, corrected C, and gave an opinion on E. That was not always correct. A citation does not mean the answer is true. It only shows which chunk was in the prompt. A real user could follow steps that do not exist, or believe that Poetry runs tests. When the chatbot cannot find reliable information, it should say clearly that it could not find it in the docs.

---

## Step 7 – Chat Page in Streamlit

**Purpose.** `pages/6_Poetry_Chat.py` lets anyone use the chatbot in the browser, with sources shown.

**How it works:**
- `st.chat_input` is the box where the user types.
- `st.chat_message` shows a message in a chat bubble (`user` or `assistant`).
- `st.session_state` keeps the messages. Streamlit re-runs the script after every action. Without it, the old messages would disappear.
- `answer_question` connects the page to the RAG system. Each question is answered alone, with no memory of earlier ones.

**Sources.** Under each answer there is one `st.expander` per source. The title shows the number, title, file and similarity. Inside I show the **title**, the **file name**, the **similarity score**, and the **chunk text**.

**Sidebar.** A slider sets `k` from 1 to 8 (default 3). The **Clear chat** button empties the messages.

**API key.**

| | Where the key is | How it is read |
|---|---|---|
| Local | `.env` file | `load_dotenv()` |
| Streamlit Cloud | App Settings → Secrets: `GROQ_API_KEY = "..."` | `st.secrets` |

If the key is missing, the page shows a message and turns off the input.

**ChromaDB collection.** The page builds it automatically on the first run (from `data/poetry_docs`) and keeps it with `st.cache_resource`. This is needed because `chroma_db/` is not on GitHub, so a new server does not have it.

**Deployment:**
1. Push the code to GitHub (no key, no `.venv`, no `chroma_db`).
2. In Streamlit Cloud, create the app from the repo.
3. Add `GROQ_API_KEY` in Secrets.
4. Open the live link and test.

**Testing so far:**
- Local: the page runs. The question "How do I remove a package from a dependency group?" gave a right answer with sources. ==[Fill in: results of the other test questions, the slider and Clear chat]==
- Online: ==[Fill in: deployment result]==. My earlier deployment crashed because of `torch` (from `sentence-transformers`). If it fails again, I send the logs to Mohammad.

### Security

Never push these to GitHub:
- **`.env` and API keys:** anyone with my key can use my Groq account. Bots scan GitHub for keys. If a key reaches the git history, I must make a new one.
- **`.venv`:** it is big, it belongs to my computer, and it can be recreated.
- **`chroma_db`:** it is generated data. It is big, and the app can rebuild it.

---

## Step 8 – Tests

**Why tests?** When I change the code, tests tell me in seconds if I broke something. I do not need to run the chatbot by hand.

**Tools:**
- **`pytest`** finds functions named `test_...`, runs them, and shows which failed.
- **`unittest.mock`** replaces a real function with a fake one. I choose what it returns and I can check how it was called.

**Why fake functions?** The real `query_collection` needs a database. The real `ask_llm` needs the internet, a key and a real model. Tests should not depend on them because:
- they are slow,
- a real model does not always give the same answer,
- they cost money and need internet,
- I want to test **my code**, not the model.

**Limit:** these tests do not prove the model answers correctly. I checked that by hand in Step 4.

### Tests in `tests/test_rag.py` (16 tests)

| Function | What I test |
|---|---|
| `format_sources` | Numbers start at 1. Each source has title, file name and text. |
| `build_messages` | The system prompt is included. All sources come before the question. The question is at the end. |
| `answer_question` | `query_collection` and `ask_llm` are replaced with `unittest.mock`. The dictionary has `answer`, `sources`, `prompt`. `ask_llm` is called once with `temperature=0`. |
| Edge cases | Empty results and `k=0` do not crash. |

**A limit:** the `k=0` test uses a fake search. I did not test the real `query_collection` with `k=0`. It may reject it. The Streamlit slider starts at 1.

I also broke `rag.py` on purpose in 4 small ways. Each time, 2 to 5 tests failed. ==[Fill in: which 4 changes]==

**Results:** `python -m pytest` → 76 passed in 2.71 s (16 from `test_rag.py`). `ruff check .` → All checks passed.

---

## Overall Reflection

- **RAG:** search first, then write. If the search is bad, the model has nothing good to use.
- **Retrieved chunks matter.** With the right chunk, 5 of 5 answers were right. Without it, 2 of 4. Q1 and Q6 failed at every `k`, so the problem was retrieval, not the model.
- **Do not trust answers.** Some wrong answers looked good and had citations. Even the plain model gave old commands that looked normal.
- **Retrieval vs generation failures** need different fixes. If I mix them up, I fix the wrong part.
- **Citations help but do not prove.** 2 of 11 RAG answers had unsupported citations.
- **Real questions matter.** My special questions showed problems normal ones did not: answers from memory, opinions with citations, and a made-up guide. With only 11 questions, my comparison is weak.
- **Testing and deployment.** Tests let me trust my code without calling the model. A clean server needs the key from secrets and a rebuilt database. ==[Fill in: what I learned from the real deployment]==

**What I would improve:**
- Add my teammates' questions to reach 30 and run everything again.
- Improve chunking or ranking.
- Make the chatbot say "I could not find this in the docs" when needed.
- Check that each citation supports its sentence.

---

## Bonus (Optional)

I mark an item only if I really did it.

- [ ] Use an AI model as a judge.
- [ ] Compare two AI models.
- [ ] Measure response time and token usage.
