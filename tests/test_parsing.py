"""Strict output-format parsing tests."""

from __future__ import annotations

from ledgergym.parsing import parse_response


def test_valid_single_block():
    parsed = parse_response(
        '<reasoning>because</reasoning><answer>{"emi": 16012.45}</answer>'
    )
    assert parsed.format_valid
    assert parsed.answer == {"emi": 16012.45}
    assert parsed.reasoning == "because"
    assert parsed.n_answer_blocks == 1


def test_accepts_no_reasoning_tag():
    parsed = parse_response('<answer>{"emi": 1}</answer>')
    assert parsed.format_valid
    assert parsed.reasoning is None


def test_missing_answer_block_is_invalid():
    parsed = parse_response("<reasoning>only reasoning</reasoning>")
    assert not parsed.format_valid
    assert parsed.error == "missing_answer_block"


def test_multiple_answer_blocks_is_invalid_and_flagged():
    parsed = parse_response(
        '<answer>{"emi": 1}</answer> and also <answer>{"emi": 2}</answer>'
    )
    assert not parsed.format_valid
    assert parsed.multiple_answers
    assert parsed.error == "multiple_answer_blocks"


def test_answer_must_be_json_object():
    assert not parse_response("<answer>[1, 2, 3]</answer>").format_valid
    assert parse_response("<answer>not json</answer>").error == "answer_not_json"
    assert parse_response("<answer>42</answer>").error == "answer_not_json_object"


def test_nested_values_are_rejected():
    parsed = parse_response('<answer>{"emi": {"nested": 1}}</answer>')
    assert not parsed.format_valid
    assert parsed.error == "answer_values_not_scalar"


def test_empty_object_is_invalid():
    parsed = parse_response("<answer>{}</answer>")
    assert not parsed.format_valid


def test_char_length_is_tracked():
    text = '<answer>{"emi": 1}</answer>'
    assert parse_response(text).char_len == len(text)
