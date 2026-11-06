"""Tests for task9/code/rag.py. No network, no real model, no real database."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task9.code import rag

QUESTION = "How do I add a dev dependency?"

RESULTS = [
    {
        "text": "Use poetry add pytest --group test.",
        "metadata": {
            "title": "Managing dependencies",
            "source": "managing-dependencies.md",
        },
        "similarity": 0.81,
    },
    {
        "text": "The --dev option is a shortcut for -G dev.",
        "metadata": {"title": "Commands", "source": "cli.md"},
        "similarity": 0.64,
    },
]


def run_answer_question(results=None, answer="Use poetry add [1].", **kwargs):
    """Call answer_question with query_collection and ask_llm replaced by fakes."""
    fake_results = RESULTS if results is None else results
    fake_query = MagicMock(return_value=fake_results)
    fake_llm = MagicMock(return_value=answer)
    with (
        patch.object(rag, "query_collection", fake_query),
        patch.object(rag, "ask_llm", fake_llm),
    ):
        result = rag.answer_question(QUESTION, "poetry_800_150", **kwargs)
    return result, fake_query, fake_llm


# --- format_sources ---


def test_format_sources_numbers_start_at_one():
    text = rag.format_sources(RESULTS)
    assert text.startswith("[1] ")
    assert "[2] " in text
    assert "[0]" not in text
    assert "[3]" not in text


def test_format_sources_has_title_file_and_text_for_each_source():
    text = rag.format_sources(RESULTS)
    for result in RESULTS:
        assert f"Title: {result['metadata']['title']}" in text
        assert f"File: {result['metadata']['source']}" in text
        assert f"Text: {result['text']}" in text


def test_format_sources_keeps_each_source_with_its_own_number():
    text = rag.format_sources(RESULTS)
    first, second = text.split("\n\n")
    assert first.startswith("[1] Title: Managing dependencies")
    assert "cli.md" not in first
    assert second.startswith("[2] Title: Commands")


def test_format_sources_empty_list_gives_empty_text():
    assert rag.format_sources([]) == ""


# --- build_messages ---


def test_build_messages_returns_the_system_prompt():
    system_prompt, _ = rag.build_messages(QUESTION, RESULTS)
    assert system_prompt == rag.SYSTEM_PROMPT


def test_build_messages_puts_all_sources_before_the_question():
    _, user_message = rag.build_messages(QUESTION, RESULTS)
    question_position = user_message.index("Question:")
    for result in RESULTS:
        assert user_message.index(result["text"]) < question_position
    assert user_message.startswith("Sources:\n")


def test_build_messages_ends_with_the_question():
    _, user_message = rag.build_messages(QUESTION, RESULTS)
    assert user_message.endswith(f"Question: {QUESTION}")


def test_build_messages_with_no_results_still_ends_with_the_question():
    _, user_message = rag.build_messages(QUESTION, [])
    assert user_message.endswith(f"Question: {QUESTION}")


# --- answer_question ---


def test_answer_question_returns_answer_sources_and_prompt():
    result, _, _ = run_answer_question(embed_model=object())
    assert set(result) == {"answer", "sources", "prompt"}
    assert result["answer"] == "Use poetry add [1]."
    assert result["sources"] == [
        {
            "title": "Managing dependencies",
            "source": "managing-dependencies.md",
            "similarity": 0.81,
            "text": "Use poetry add pytest --group test.",
        },
        {
            "title": "Commands",
            "source": "cli.md",
            "similarity": 0.64,
            "text": "The --dev option is a shortcut for -G dev.",
        },
    ]
    assert result["prompt"].startswith(f"System:\n{rag.SYSTEM_PROMPT}")
    assert "[1] Title: Managing dependencies" in result["prompt"]
    assert result["prompt"].endswith(f"Question: {QUESTION}")


def test_answer_question_calls_ask_llm_once_with_temperature_zero():
    _, _, fake_llm = run_answer_question(embed_model=object())
    fake_llm.assert_called_once()
    args, kwargs = fake_llm.call_args
    assert kwargs["temperature"] == 0
    assert args[0] == rag.SYSTEM_PROMPT
    assert QUESTION in args[1]
    assert args[2] == rag.LLM_MODEL


def test_answer_question_passes_a_custom_llm_model():
    _, _, fake_llm = run_answer_question(embed_model=object(), llm_model="some-model")
    assert fake_llm.call_args.args[2] == "some-model"


def test_answer_question_asks_the_store_for_k_chunks():
    embed_model = object()
    _, fake_query, _ = run_answer_question(embed_model=embed_model, k=2)
    fake_query.assert_called_once_with("poetry_800_150", QUESTION, embed_model, k=2)


def test_answer_question_default_k_is_three():
    _, fake_query, _ = run_answer_question(embed_model=object())
    assert fake_query.call_args.kwargs["k"] == 3


def test_answer_question_loads_the_embedding_model_when_none_is_given():
    fake_model = object()
    with patch.object(
        rag, "_load_embedding_model", return_value=fake_model
    ) as fake_loader:
        _, fake_query, _ = run_answer_question()
    fake_loader.assert_called_once()
    assert fake_query.call_args.args[2] is fake_model


# --- empty results / k=0 ---


def test_answer_question_with_empty_results_does_not_crash():
    result, _, fake_llm = run_answer_question(results=[], embed_model=object())
    assert result["sources"] == []
    assert result["answer"] == "Use poetry add [1]."
    fake_llm.assert_called_once()
    assert result["prompt"].endswith(f"Question: {QUESTION}")


def test_answer_question_with_k_zero_does_not_crash():
    result, fake_query, _ = run_answer_question(results=[], embed_model=object(), k=0)
    assert fake_query.call_args.kwargs["k"] == 0
    assert result["sources"] == []