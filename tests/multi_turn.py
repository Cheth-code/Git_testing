import yaml
import pytest
from collections import defaultdict

from rag_app import langsmith_rag

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric


# -------------------------------------------------------------------
# Load test cases from YAML
# -------------------------------------------------------------------
def load_cases():
    with open("tests/test_data.yaml") as f:
        return yaml.safe_load(f)["rag_unit_tests"]


# -------------------------------------------------------------------
# In-memory conversation store (session simulation)
# Keyed by conversation_key
# -------------------------------------------------------------------
conversation_memory = defaultdict(list)


# -------------------------------------------------------------------
# Smoke test (single-turn sanity)
# -------------------------------------------------------------------
def test_rag_returns_answer():
    answer = langsmith_rag("What is FinOps?")
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0


# -------------------------------------------------------------------
# Multi-turn RAG unit test with DeepEval
# -------------------------------------------------------------------
@pytest.mark.parametrize(
    "case",
    load_cases(),
    ids=lambda c: c["id"],
)
def test_rag_multi_turn_answer_quality(case):

    query = case["input_query"]
    key = case.get("conversation_key")
    turn_index = case.get("turn_index", 1)

    # ---------------------------------------------------------------
    # Build conversation context (previous turns)
    # ---------------------------------------------------------------
    messages = []

    if key and turn_index > 1:
        messages.extend(conversation_memory[key])

    messages.append({"role": "user", "content": query})

    # ---------------------------------------------------------------
    # Call RAG (black-box)
    # ---------------------------------------------------------------
    answer = langsmith_rag(query)

    assert isinstance(answer, str)
    assert len(answer.strip()) > 0

    # ---------------------------------------------------------------
    # Persist conversation for next turn
    # ---------------------------------------------------------------
    if key:
        conversation_memory[key].extend(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer},
            ]
        )

    # ---------------------------------------------------------------
    # DeepEval quality check (per turn)
    # ---------------------------------------------------------------
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        expected_output=case["ground_truth_answer"],
    )

    metric = AnswerRelevancyMetric(
        threshold=case["evaluation"]["threshold"]
    )

    assert_test(test_case, [metric])
