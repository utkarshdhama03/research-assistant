import uuid

import pytest

from research_assistant.llm.aliases import AliasMap
from research_assistant.llm.errors import UnknownAliasError

IDS = [uuid.uuid4() for _ in range(12)]


def test_round_trip_in_order() -> None:
    aliases = AliasMap(IDS)
    assert [aliases.alias(cid) for cid in IDS] == [f"C{i}" for i in range(1, 13)]
    assert [aliases.resolve(f"C{i}") for i in range(1, 13)] == IDS
    assert aliases.resolve_all(["C3", "C12", "C1"]) == [IDS[2], IDS[11], IDS[0]]
    assert len(aliases) == 12 and aliases.items()[0] == ("C1", IDS[0])


def test_surrounding_whitespace_is_ignored() -> None:
    assert AliasMap(IDS).resolve("  C2\n") == IDS[1]


@pytest.mark.parametrize(
    "bad", ["C0", "C13", "C99", "c1", "C01", "[C1]", "C1,C2", "C 1", "", "1", str(IDS[0])]
)
def test_unknown_alias_is_an_error(bad: str) -> None:
    with pytest.raises(UnknownAliasError):
        AliasMap(IDS).resolve(bad)


def test_unknown_alias_in_a_list_fails_the_whole_list() -> None:
    with pytest.raises(UnknownAliasError, match="C40"):
        AliasMap(IDS).resolve_all(["C1", "C40"])


def test_unknown_chunk_id() -> None:
    with pytest.raises(LookupError):
        AliasMap(IDS).alias(uuid.uuid4())


def test_duplicate_chunk_ids_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        AliasMap([IDS[0], IDS[1], IDS[0]])


def test_empty_map() -> None:
    with pytest.raises(UnknownAliasError):
        AliasMap([]).resolve("C1")


def test_custom_prefix_and_generic_keys() -> None:
    urls = ["https://a.example/", "https://b.example/"]
    aliases = AliasMap(urls, prefix="R")
    assert aliases.aliases == ["R1", "R2"] and aliases.resolve("R2") == urls[1]
    with pytest.raises(UnknownAliasError, match="R1..R2"):
        aliases.resolve("C1")


@pytest.mark.parametrize("prefix", ["", "c", "CC", "1"])
def test_bad_prefix(prefix: str) -> None:
    with pytest.raises(ValueError):
        AliasMap(IDS, prefix=prefix)
