from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.hello_world.llm import ask_llm, parse_label


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"label": "positive"}', "positive"),
        ('{"label": "  NeGaTiVe  "}', "negative"),
        ('Sure! {"label": "neutral"} Hope that helps.', "invalid"),
        ('{"label": "happy"}', "invalid"),
        ("", "invalid"),
        (None, "invalid"),
        ('["positive"]', "invalid"),
        ('{"label": 5}', "invalid"),
    ],
)
def test_parse_label(text, expected):
    assert parse_label(text) == expected


def test_ask_llm_uses_fake_client(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")

    fake_reply = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"label": "positive"}'))]
    )

    with patch("src.hello_world.llm.Groq") as fake_groq:
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_reply
        fake_groq.return_value = fake_client

        result = ask_llm("sys prompt", "nice app", "some-model")

    assert result == '{"label": "positive"}'
    fake_groq.assert_called_once_with(api_key="fake-key")
    fake_client.chat.completions.create.assert_called_once_with(
        model="some-model",
        messages=[
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "nice app"},
        ],
        temperature=0,
    )