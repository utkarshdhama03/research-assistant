"""The evaluation question set (questions.yaml next to this module)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

QUESTIONS_FILE = Path(__file__).parent / "questions.yaml"
CATEGORIES = frozenset({"technical", "science", "history-events", "india", "contested", "false-premise"})


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    category: str
    question: str
    expectation: str | None = None


def load_questions(path: Path = QUESTIONS_FILE) -> list[EvalQuestion]:
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: expected a non-empty 'questions' list")
    out: list[EvalQuestion] = []
    for item in items:
        if not isinstance(item, dict) or not all(
            isinstance(item.get(k), str) for k in ("id", "category", "question")
        ):
            raise ValueError(f"{path}: every question needs string id, category and question: {item!r}")
        if item["category"] not in CATEGORIES:
            raise ValueError(f"{path}: unknown category {item['category']!r} for {item['id']}")
        expectation = item.get("expectation")
        out.append(EvalQuestion(item["id"], item["category"], item["question"], expectation))
    ids = [q.id for q in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate question ids")
    return out
