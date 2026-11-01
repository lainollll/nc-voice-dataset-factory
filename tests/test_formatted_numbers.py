from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "reference,hypothesis",
    [
        (
            "Sarah counted three thousand, six hundred and nineteen chairs before opening the meeting room.",
            "Sarah counted 3,619 chairs before opening the meeting room.",
        ),
        (
            "The distance was four thousand, four hundred and sixty-four metres.",
            "The distance was 4,464 meters.",
        ),
        (
            "The shipment weighed three thousand, one hundred and ninety-nine kilograms.",
            "The shipment weighed 3,199 kg.",
        ),
        ("The parcel weighs one kilogram.", "The parcel weighs 1 kg."),
    ],
)
def test_asr_formatted_integer_and_unit_conventions_are_equivalent(reference, hypothesis):
    from nc_factory.transcript import transcript_metrics

    metrics = transcript_metrics(reference, hypothesis, "en")
    assert metrics["wer"] == 0 and metrics["cer"] == 0


def test_formatted_number_comparison_retains_wrong_numbers_and_ambiguous_forms():
    from nc_factory.transcript import normalize_for_comparison, transcript_metrics

    assert (
        transcript_metrics(
            "There were three thousand, six hundred and nineteen chairs.", "There were 3,620 chairs."
        )["wer"]
        > 0
    )
    assert "3" in normalize_for_comparison("The value is 3,619.5.", "en").split()
    assert "3" in normalize_for_comparison("Värdet är 3,619.", "sv").split()
    assert "3" in normalize_for_comparison("The range is 3,619-5,000.", "en").split()


def test_sentence_final_integer_expands_without_guessing_decimals_or_dates():
    from nc_factory.text import normalize_text
    from nc_factory.transcript import transcript_metrics

    assert normalize_text("Please read chapter 20.") == "Please read chapter twenty."
    assert normalize_text("There are 20, and I counted them.") == "There are twenty, and I counted them."
    assert "3.14" in normalize_text("The value is 3.14.")
    assert "2026-10-31" in normalize_text("The date is 2026-10-31.")
    assert transcript_metrics("Please read chapter twenty.", "Please read chapter 20.")["wer"] == 0
