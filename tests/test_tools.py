"""Tests for the optional calculator tool and formatting helpers."""

from __future__ import annotations

import pytest

from ledgergym.fmt import format_inr, parse_number
from ledgergym.tools.calculator import CalculatorError, CalculatorTool, safe_eval


def test_format_inr_indian_grouping():
    assert format_inr(500000) == "5,00,000"
    assert format_inr(12000000) == "1,20,00,000"
    assert format_inr(999) == "999"
    assert format_inr(16012.45, decimals=2) == "16,012.45"


def test_parse_number_handles_grouping_and_symbols():
    assert parse_number("5,00,000") == 500000.0
    assert parse_number("₹16,012.45") == pytest.approx(16012.45)
    assert parse_number(1234) == 1234.0
    assert parse_number("not a number") is None
    assert parse_number(True) is None
    assert parse_number(None) is None


def test_safe_eval_arithmetic():
    assert safe_eval("2 + 3 * 4") == 14.0
    assert safe_eval("(1 + 1) ** 10") == 1024.0
    assert safe_eval("-5 % 3") == pytest.approx(1.0)


def test_safe_eval_rejects_code_execution():
    with pytest.raises(CalculatorError):
        safe_eval("__import__('os').system('echo hi')")
    with pytest.raises(CalculatorError):
        safe_eval("open('x')")


def test_calculator_tool_returns_json_result():
    tool = CalculatorTool()
    assert tool("3 * 7")["result"] == 21.0
    failed = tool("this is not math")
    assert failed["ok"] is False
    assert "error" in failed
