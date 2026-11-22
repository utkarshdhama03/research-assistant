"""Numbers in claims and quotes (pure). A claim's numbers must all occur in its quote.

Recognized: integers, decimals, percentages, years, ranges ("1.8–2.8", "1.8-2.8", "1.8 to 2.8",
"-15°F to -25°F", "20-30%"), ordinals ("23rd" -> 23). Thousands separators are removed ("1,000" ->
1000; "23, 2023" stays two numbers) and en/em dashes become "-". Values compare as decimals (2.0 == 2).

Identifiers are not numbers: a number right after a letter, with or without a hyphen ("Chandrayaan-2",
"GPT-4", "COVID-19", "H2O", "qwen3", "H100"), or with a letter right after it ("7B", "13b", "2xl",
"LLaMA2-7B"). Exceptions, still numbers: ordinals and decades ("23rd", "1990s"), multipliers ("10x"), and
the units in `_UNIT_SUFFIXES` ("128km/h", "10kg", "3.5GHz", "16GB"); any other glued unit is skipped.

Limits (decision log): the sign is ignored (a leading dash is ambiguous between a minus and a range
dash), spelled-out numbers ("three") are not recognized.
"""

import re
from dataclasses import dataclass
from decimal import Decimal

_DASHES = str.maketrans({"–": "-", "—": "-", "‒": "-", "−": "-", "‐": "-", "‑": "-"})
_THOUSANDS = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])")
_NUM = r"\d+(?:\.\d+)?"
# Not part of a longer number, and not an identifier: no letter right before, with or without a hyphen.
_START = r"(?<![A-Za-z0-9.])(?<![A-Za-z]-)"
# Letters glued to the end of a number that keep it a number; any other letter makes it an identifier.
_UNIT_SUFFIXES = (
    "st|nd|rd|th|s|x|km|kg|mg|mm|cm|kw|kwh|mw|mwh|gw|gwh|tw|twh|hz|khz|mhz|ghz|mph|kph|kb|mb|gb|tb|ms"
)
# (?!\.?\d): the whole number, so backtracking cannot cut "13b" to "1" or "1.5B" to "1".
_END = rf"(?!\.?\d)(?=(?i:{_UNIT_SUFFIXES})?(?![A-Za-z]))"
_UNIT = r"(?:\s?(?:%|percent\b|°\s?[CFcf]))?"
_RANGE = re.compile(rf"{_START}({_NUM}){_END}{_UNIT}\s*(?:-|\bto\b)\s*-?\s*({_NUM}){_END}")
_SINGLE = re.compile(rf"{_START}({_NUM}){_END}")

Range = tuple[Decimal, Decimal]


@dataclass(frozen=True)
class NumberSet:
    singles: frozenset[Decimal]
    ranges: frozenset[Range]

    @property
    def values(self) -> frozenset[Decimal]:
        """Singles and range endpoints."""
        return self.singles | {v for r in self.ranges for v in r}


def _normalize(text: str) -> str:
    text = text.translate(_DASHES)
    return _THOUSANDS.sub(lambda m: m.group().replace(",", ""), text)


def _range(a: str, b: str) -> Range:
    x, y = Decimal(a), Decimal(b)
    return (x, y) if x <= y else (y, x)


def extract_numbers(text: str) -> NumberSet:
    normalized = _normalize(text)
    ranges = set()
    spans = []
    for m in _RANGE.finditer(normalized):
        ranges.add(_range(m.group(1), m.group(2)))
        spans.append((m.start(), m.end()))
    rest = list(normalized)
    for start, end in spans:  # numbers inside a range are not singles
        rest[start:end] = " " * (end - start)
    singles = {Decimal(m.group(1)) for m in _SINGLE.finditer("".join(rest))}
    return NumberSet(frozenset(singles), frozenset(ranges))


def _fmt(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def missing_numbers(claim: str, quote: str, exempt: frozenset[Decimal] = frozenset()) -> list[str]:
    """Numbers of the claim that the quote does not contain. A claim range must equal a quote range
    (overlap is not enough); a single claim number may occur in the quote alone or as a range end.
    Numbers in `exempt` (e.g. those of the research question) are never missing; a range is exempt
    when both its ends are."""
    c, q = extract_numbers(claim), extract_numbers(quote)
    missing = [
        f"{_fmt(lo)}-{_fmt(hi)}"
        for lo, hi in sorted(c.ranges)
        if (lo, hi) not in q.ranges and not (lo in exempt and hi in exempt)
    ]
    missing += [_fmt(v) for v in sorted(c.singles) if v not in q.values and v not in exempt]
    return missing
