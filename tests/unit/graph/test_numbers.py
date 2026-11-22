import json
from decimal import Decimal
from pathlib import Path

import pytest

from research_assistant.graph.numbers import extract_numbers, missing_numbers

HEAT_PUMP_QUOTE = "At -15°C, that typically drops to 1.8–2.8 depending on design."


def test_heat_pump_claim_with_a_different_range_is_caught() -> None:
    # Live run 344b505b, claim 6: the quote says 1.8–2.8, the claim said 2.0–2.8.
    claim = "At -15°C, cold-climate heat pumps (CCHPs) typically achieve a COP of 2.0–2.8."
    assert missing_numbers(claim, HEAT_PUMP_QUOTE) == ["2-2.8"]
    # Claim 7: a range the quote does not contain at all.
    assert missing_numbers("At -15°C, standard ASHPs achieve a COP of 1.5–1.9.", HEAT_PUMP_QUOTE) == [
        "1.5-1.9"
    ]


@pytest.mark.parametrize(
    "claim", ["COP of 1.8–2.8 at -15°C", "COP of 1.8-2.8 at -15 °C", "COP 1.8 to 2.8 at 15°C"]
)
def test_same_range_in_any_notation_matches(claim: str) -> None:
    assert missing_numbers(claim, HEAT_PUMP_QUOTE) == []


def test_overlapping_range_is_not_enough() -> None:
    assert missing_numbers("between 1.8 and 2.5", "between 1.8–2.8") == ["2.5"]
    assert missing_numbers("from 1.8–2.5", "from 1.8–2.8") == ["1.8-2.5"]


def test_dates_in_different_formats_match() -> None:
    assert (
        missing_numbers("It landed on August 23, 2023.", "The lander touched down on 23rd August 2023.") == []
    )
    assert missing_numbers(
        "It landed on August 24, 2023.", "The lander touched down on 23rd August 2023."
    ) == ["24"]


def test_thousands_separators_and_lists() -> None:
    assert missing_numbers("about 1000 people", "about 1,000 people") == []
    assert missing_numbers("55,000,000 people", "55000000 people") == []
    assert extract_numbers("on August 23, 2023").singles == {Decimal(23), Decimal(2023)}  # not 23,2023


def test_percentages_years_decimals() -> None:
    quote = "In 2019, capacity grew by 20-30% and efficiency by 4.5 percent."
    assert missing_numbers("In 2019 capacity grew 20–30%.", quote) == []
    assert missing_numbers("Efficiency rose 4.5% in 2019.", quote) == []
    assert missing_numbers("Efficiency rose 4.6% in 2020.", quote) == ["4.6", "2020"]
    assert missing_numbers("capacity grew 25%", quote) == ["25"]  # inside the range is not the same


def test_identifiers_and_hyphenated_numbers() -> None:
    # Identifiers (a number right after a letter, with or without a hyphen) are not numbers.
    assert missing_numbers("Chandrayaan-2's lander failed.", "Chandrayaan-3 landed.") == []
    assert extract_numbers("GPT-4, COVID-19, H2O, qwen3 and Chandrayaan-2").values == frozenset()
    # Units after a number do not make it an identifier.
    assert extract_numbers("128km/h and 10kg").singles == {Decimal(128), Decimal(10)}
    assert missing_numbers("qwen3 and B2B sales", "no numbers here") == []
    assert extract_numbers("1990s, the 4th, 10x faster, 3.5GHz, 16GB").singles == {
        Decimal(1990),
        Decimal(4),
        Decimal(10),
        Decimal("3.5"),
        Decimal(16),
    }


@pytest.mark.parametrize("text", ["LLaMA2-7B", "7B", "13b", "1.5B", "2xl", "H100", "a 70b model"])
def test_numbers_glued_to_letters_are_identifiers(text: str) -> None:
    assert extract_numbers(text).values == frozenset()


def test_final_eval_context_numbers() -> None:
    # Final eval: "LLaMA2-7B" was read as the number 7; numbers standing alone are still checked.
    assert missing_numbers("LLaMA2-7B was fine-tuned on the data.", "The model was fine-tuned.") == []
    assert missing_numbers("The company had 150 employees.", "The company was small.") == ["150"]
    assert missing_numbers("The 2015 El Niño was strong.", "El Niño was strong.") == ["2015"]
    assert extract_numbers("-15°F to -25°F (-26°C to -32°C)").ranges == {
        (Decimal(15), Decimal(25)),
        (Decimal(26), Decimal(32)),
    }


def test_decimal_equality_and_endpoints() -> None:
    assert missing_numbers("COP 2.0", "COP 2") == []
    assert missing_numbers("up to 2.8", HEAT_PUMP_QUOTE) == []  # a single number may be a range end
    assert missing_numbers("a claim with no numbers", "1, 2, 3") == []


LIVE = json.loads(
    (Path(__file__).parents[2] / "fixtures" / "number_check_cases.json").read_text(encoding="utf-8")
)["cases"]


@pytest.mark.parametrize("case", LIVE, ids=lambda c: f"{c['eval_question']}-{c['claim'][:30]}")
def test_live_number_cases_with_exemptions(case: dict[str, str]) -> None:
    """quality2 eval: 4 strict rejections now pass, the 2 true catches are still rejected."""
    exempt = extract_numbers(f"{case['question']} {case['sub_question']}").values
    assert missing_numbers(case["claim"], case["quote"], exempt) == case["expected_missing"]


def test_live_number_case_split() -> None:
    assert sum(1 for c in LIVE if c["expected_missing"]) == 2 and len(LIVE) == 6


def test_exempt_numbers() -> None:
    exempt = extract_numbers("Why did the 2008 financial crisis start?").values
    assert missing_numbers("The 2008 crisis began in housing.", "The crisis began in housing.", exempt) == []
    assert missing_numbers("The 2009 recession followed.", "The recession followed.", exempt) == ["2009"]
    assert (
        missing_numbers("From 2008-2010 it grew.", "It grew.", extract_numbers("2008 and 2010").values) == []
    )
