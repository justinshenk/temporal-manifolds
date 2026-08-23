"""Tests for the complete abstract-prompt generator."""

from __future__ import annotations


def test_abstract_prompt_records_include_all_number_and_unit_variants() -> None:
    from temporal_manifolds.dataset import abstract
    from temporal_manifolds.dataset.generate_abstract import generate_abstract_prompt_records
    from temporal_manifolds.dataset.utils import (
        NUMBER_FORMATS,
        number_to_words,
        render_unit,
        smaller_unit_value,
    )

    records = generate_abstract_prompt_records()
    variants_per_value = sum(
        len(NUMBER_FORMATS) * (1 + (smaller_unit_value(1, unit) is not None))
        for unit in abstract.time_units
    )
    expected_record_count = (
        len(abstract.templates)
        * len(abstract.tasks)
        * len(abstract.values)
        * variants_per_value
    )

    assert len(records) == expected_record_count
    base_value = 2
    base_unit = "hours"
    smaller = smaller_unit_value(base_value, base_unit)
    assert smaller is not None
    smaller_value, smaller_unit = smaller

    sample_records = [
        record
        for record in records
        if record["template_id"] == "task_available_time"
        and record["task"] == "allocate limited resources among competing needs"
        and record["base_value"] == base_value
        and record["base_unit"] == base_unit
    ]
    rendered_variants = {
        (
            record["unit_variant"],
            record["number_format"],
            record["value_text"],
            record["unit"],
        )
        for record in sample_records
    }

    assert rendered_variants == {
        ("original", "numeric", str(base_value), render_unit(base_value, base_unit)),
        ("original", "words", number_to_words(base_value), render_unit(base_value, base_unit)),
        ("smaller", "numeric", str(smaller_value), render_unit(smaller_value, smaller_unit)),
        (
            "smaller",
            "words",
            number_to_words(smaller_value),
            render_unit(smaller_value, smaller_unit),
        ),
    }


def test_abstract_prompt_text_projection_preserves_order_and_duplicates(monkeypatch) -> None:
    from temporal_manifolds.dataset import generate_abstract as generate_abstract_module

    records = [{"text": "first"}, {"text": "second"}, {"text": "first"}]
    monkeypatch.setattr(
        generate_abstract_module,
        "generate_abstract_prompt_records",
        lambda **_kwargs: records,
    )

    assert generate_abstract_module.generate_abstract_prompts() == [
        "first",
        "second",
        "first",
    ]
