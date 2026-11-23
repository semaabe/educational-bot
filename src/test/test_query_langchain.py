"""Regression coverage for Task 8: one LLM invocation per response."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from services import queryLangchain


@pytest.fixture
def llm(monkeypatch):
    client = Mock(spec=["invoke"])
    client.invoke.return_value = SimpleNamespace(content="Generated answer")
    monkeypatch.setattr(queryLangchain, "create_llm", lambda: client)
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data, expected_data",
    [
        (None, "[]"),
        (["Course notes"], "['Course notes']"),
        ({"topic": "Robotics"}, "{'topic': 'Robotics'}"),
    ],
    ids=["no-context", "list-context", "dict-context"],
)
async def test_plain_role_invokes_llm_once(llm, data, expected_data):
    result = await queryLangchain.fetchGptResponse(
        "Explain the topic", "You are a tutor.", data
    )

    llm.invoke.assert_called_once_with(
        [
            ("system", f"You are a tutor. Here are the relevant information {expected_data}."),
            ("user", "Explain the topic"),
        ]
    )
    assert result == "Generated answer"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [
        "Question: {{ query }}\nContext: {{ relevant_documents }}",
        "Question: {query}\nContext: {relevant_documents}",
    ],
    ids=["jinja-template", "format-template"],
)
async def test_only_rendered_and_trimmed_context_is_sent(llm, role):
    documents = ["A" * 1600, "Second document", "Third document", "Excluded fourth document"]
    data = {"relevant_documents": documents}
    original_data = deepcopy(data)

    result = await queryLangchain.fetchGptResponse("Explain the topic", role, data)

    expected_documents = ["A" * 1500 + "...", "Second document", "Third document"]
    llm.invoke.assert_called_once_with(
        [
            ("system", f"Question: Explain the topic\nContext: {expected_documents}"),
            ("user", "Explain the topic"),
        ]
    )
    assert result == "Generated answer"
    assert data == original_data


@pytest.mark.asyncio
async def test_grading_aliases_are_rendered_before_single_call(llm):
    result = await queryLangchain.fetchGptResponse(
        "Grade this answer",
        "Submission: {{ submission }}\nRubric: {{ rubric }}",
        {
            "student_submission": "My answer",
            "retrieved_context": ["Award points for correct reasoning"],
        },
    )

    llm.invoke.assert_called_once_with(
        [
            ("system", "Submission: My answer\nRubric: ['Award points for correct reasoning']"),
            ("user", "Grade this answer"),
        ]
    )
    assert result == "Generated answer"


@pytest.mark.asyncio
async def test_llm_error_propagates_without_another_invocation(llm):
    llm.invoke.side_effect = RuntimeError("LLM unavailable")

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        await queryLangchain.fetchGptResponse("Explain the topic", "You are a tutor.")

    llm.invoke.assert_called_once()
