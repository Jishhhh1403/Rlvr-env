"""Formatting helpers: Indian digit grouping and tolerant numeric parsing."""

from __future__ import annotations

import re

__all__ = ["format_inr", "parse_number"]


def format_inr(value: float | int, *, decimals: int | None = None) -> str:
    """Format an integer amount using Indian digit grouping (5,00,000 / 1,20,00,000).

    If ``decimals`` is given, the value is first rounded to that many decimals
    and rendered with the fractional part after a decimal point.
    """
    if decimals is not None:
        value = round(float(value), decimals)
        int_part = int(abs(value))
        frac = f"{abs(value) - int_part:.{decimals}f}"[1:] if decimals else ""
    else:
        if abs(value - round(value)) > 1e-9:
            raise ValueError("format_inr without decimals requires an integral value")
        int_part = int(round(abs(value)))
        frac = ""
    sign = "-" if value < 0 else ""
    s = str(int_part)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups: list[str] = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    return f"{sign}{s}{frac}"


_NUMBER_RE = re.compile(r"^[+-]?[0-9][0-9,]*\.?[0-9]*$")


def parse_number(value: object) -> float | None:
    """Parse a model-emitted numeric value.

    Accepts ints/floats directly and strings with Indian or western digit
    grouping (``"5,00,000"``), trailing/leading whitespace, and a bare
    currency symbol. Returns ``None`` if the value is not numeric.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().replace("₹", "").replace("Rs.", "").replace("rs", "").strip()
    text = text.replace(" ", "")
    if not _NUMBER_RE.match(text):
        return None
    text = text.replace(",", "")
    if text in {"", "+", "-", ".", "+.", "-."}:
        return None
    try:
        return float(text)
    except ValueError:
        return None
