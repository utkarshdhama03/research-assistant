"""Short aliases (C1, C2, ... or R1, R2, ...) for IDs shown in prompts. The model never writes IDs."""

import re
from collections.abc import Hashable, Iterable, Sequence

from research_assistant.llm.errors import UnknownAliasError

_PREFIX = re.compile(r"[A-Z]")


class AliasMap[K: Hashable]:
    """Maps IDs to aliases <prefix>1..<prefix>n in order, and back. Resolution is exact."""

    def __init__(self, ids: Sequence[K], *, prefix: str = "C") -> None:
        if not _PREFIX.fullmatch(prefix):
            raise ValueError(f"alias prefix must be one capital letter, got {prefix!r}")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate ids in AliasMap")
        self.prefix = prefix
        self._to_alias = {key: f"{prefix}{i}" for i, key in enumerate(ids, start=1)}
        self._to_id = {alias: key for key, alias in self._to_alias.items()}

    def __len__(self) -> int:
        return len(self._to_alias)

    @property
    def aliases(self) -> list[str]:
        return list(self._to_id)

    def alias(self, key: K) -> str:
        try:
            return self._to_alias[key]
        except KeyError:
            raise LookupError(f"{key!r} is not in this AliasMap") from None

    def resolve(self, alias: str) -> K:
        """Exact alias only (surrounding whitespace ignored). Anything else is an error, never a guess."""
        key = self._to_id.get(alias.strip())
        if key is None:
            valid = f"{self.prefix}1..{self.prefix}{len(self)}" if len(self) else "none"
            raise UnknownAliasError(f"unknown alias {alias!r}; valid: {valid}")
        return key

    def resolve_all(self, aliases: Iterable[str]) -> list[K]:
        return [self.resolve(a) for a in aliases]

    def items(self) -> list[tuple[str, K]]:
        return list(self._to_id.items())
