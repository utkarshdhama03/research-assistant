"""Event names: the ones the code can emit (from src/) and the ones the UI listens to (frontend/).

The server sends named SSE events (`event: <name>`), and EventSource only delivers names it has a
listener for, so frontend/src/api/eventNames.ts must list every name the code can emit.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_LIST = ROOT / "frontend" / "src" / "api" / "eventNames.ts"

_NAME = r'"([a-z][a-z0-9_]*)"'
_PATTERNS = [
    re.compile(r"(?:ctx|self)\.event\(\s*" + _NAME),  # ctx.event("x", ...), also split over lines
    re.compile(r"append_event\(\s*[^,()]+,\s*[^,()]+,\s*\"[^\"]*\",\s*" + _NAME),  # 4th argument
    re.compile(r"on_event\(\s*\"[^\"]*\",\s*" + _NAME),
    re.compile(r"^[A-Z_]*EVENT\s*=\s*" + _NAME, re.MULTILINE),  # CLAIM_EVENT = "worker_claimed"
    re.compile(r"_end_before_running\([^)]*,\s*" + _NAME + r"\s*\)"),  # its event argument
]


def emitted_event_names() -> set[str]:
    names: set[str] = set()
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in _PATTERNS:
            names.update(pattern.findall(text))
    return names


def frontend_event_names() -> set[str]:
    return set(re.findall(_NAME, FRONTEND_LIST.read_text(encoding="utf-8")))
