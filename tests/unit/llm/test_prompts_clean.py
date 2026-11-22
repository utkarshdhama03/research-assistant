"""No prompt may contain an eval question or its distinctive terms: the eval must measure the system, not
examples written for its own questions."""

import re

import pytest

from research_assistant.evals.questions import load_questions
from research_assistant.llm.prompt_loader import PROMPTS_DIR

# Distinctive terms of the 16 eval questions, plus the held-out false-premise question used in live checks.
EVAL_TERMS = [
    "pgvector", "HNSW", "IVFFlat", "Chandrayaan", "MVCC", "monsoon", "El Niño", "El Nino", "UPI", "LoRA",
    "fine-tuning", "financial crisis", "subprime", "mRNA", "intermittent fasting", "James Webb", "JWST",
    "Balasore", "Coromandel", "Odisha", "borrow checker", "data races", "remote work", "GST", "Titan",
    "submersible", "OceanGate", "Eiffel", "Lyon", "Apollo",
]  # fmt: skip
PROMPTS = sorted(PROMPTS_DIR.glob("*.md"))


def test_there_are_prompts() -> None:
    assert {p.stem for p in PROMPTS} >= {"plan", "select", "analyze", "verify", "gap", "write_section"}


@pytest.mark.parametrize("path", PROMPTS, ids=lambda p: p.stem)
def test_prompt_has_no_eval_question_or_term(path: object) -> None:
    text = path.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    for q in load_questions():
        assert q.question.lower() not in text.lower(), q.id
    found = [t for t in EVAL_TERMS if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text, flags=re.IGNORECASE)]
    assert found == [], f"{path}: {found}"
