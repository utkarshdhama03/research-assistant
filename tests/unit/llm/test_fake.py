import json

import pytest

from research_assistant.llm.client import LLMCallRecord, LLMClient
from research_assistant.llm.errors import LLMOutputInvalid, LLMUnavailable
from research_assistant.llm.fake import FakeLLM
from research_assistant.llm.prompt_loader import load_prompt
from research_assistant.llm.schemas import ResearchPlan

PLAN = {"sub_questions": [{"question": f"Question number {i}?", "search_queries": ["q x"]} for i in range(3)]}
PROMPT = load_prompt("plan").render(question="q?")


def test_fake_satisfies_the_protocol() -> None:
    client: LLMClient = FakeLLM()
    assert client.warm_up().outcome == "ok"


def test_scripted_replies_in_order() -> None:
    records: list[LLMCallRecord] = []
    model = ResearchPlan.model_validate(PLAN)
    fake = FakeLLM([json.dumps(PLAN), model, "free text"], on_call=records.append)

    assert fake.generate_structured(PROMPT, ResearchPlan, num_predict=500).value == model
    assert fake.generate_structured(PROMPT, ResearchPlan, num_predict=500).value == model
    assert fake.generate_text(PROMPT, num_predict=100).value == "free text"
    assert [(c.kind, c.schema, c.num_predict) for c in fake.calls] == [
        ("structured", ResearchPlan, 500),
        ("structured", ResearchPlan, 500),
        ("text", None, 100),
    ]
    assert [r.outcome for r in records] == ["ok", "ok", "ok"]


def test_scripted_invalid_output_fails_like_the_real_client() -> None:
    records: list[LLMCallRecord] = []
    fake = FakeLLM([json.dumps({"sub_questions": []})], on_call=records.append)
    with pytest.raises(LLMOutputInvalid):
        fake.generate_structured(PROMPT, ResearchPlan, num_predict=500)
    assert records[0].outcome == "LLMOutputInvalid"


def test_scripted_exception_is_raised_and_reported() -> None:
    records: list[LLMCallRecord] = []
    fake = FakeLLM([LLMUnavailable("down")], on_call=records.append)
    with pytest.raises(LLMUnavailable) as info:
        fake.generate_text(PROMPT, num_predict=100)
    assert records[0].outcome == "LLMUnavailable" and info.value.record is records[0]


def test_exhausted_script() -> None:
    with pytest.raises(AssertionError, match="no scripted reply"):
        FakeLLM().generate_text(PROMPT, num_predict=100)
