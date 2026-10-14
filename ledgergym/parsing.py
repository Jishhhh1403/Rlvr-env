"""Strict parser for the policy's output schema.

The policy must respond with reasoning followed by exactly one answer block::

    <reasoning> ...free text... </reasoning>
    <answer> {"emi": 16012.45} </answer>

Parsing is strict: zero or multiple ``<answer>`` blocks, non-JSON payloads, or
non-object JSON all count as format-invalid, which zeroes correctness downstream.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

__all__ = ["ParsedResponse", "parse_response"]

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)


class ParsedResponse(BaseModel):
    """Result of strictly parsing one policy response."""

    raw: str = Field(exclude=True)
    char_len: int = 0
    n_answer_blocks: int = 0
    format_valid: bool = False
    answer: dict[str, Any] | None = None
    reasoning: str | None = None
    error: str | None = None

    @property
    def multiple_answers(self) -> bool:
        return self.n_answer_blocks > 1


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def parse_response(text: str) -> ParsedResponse:
    """Parse a raw policy response into a :class:`ParsedResponse`.

    A response is format-valid iff it contains exactly one ``<answer>`` block
    whose contents parse as a JSON object with scalar (string/number/bool)
    values.
    """
    if text is None:
        text = ""
    reasoning_match = _REASONING_RE.search(text)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None

    blocks = _ANSWER_RE.findall(text)
    parsed = ParsedResponse(
        raw=text,
        char_len=len(text),
        n_answer_blocks=len(blocks),
        reasoning=reasoning,
    )

    if len(blocks) == 0:
        parsed.error = "missing_answer_block"
        return parsed
    if len(blocks) > 1:
        parsed.error = "multiple_answer_blocks"
        return parsed

    payload = blocks[0].strip()
    try:
        data: Any = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        parsed.error = "answer_not_json"
        return parsed

    if not isinstance(data, dict):
        parsed.error = "answer_not_json_object"
        return parsed
    if not all(isinstance(k, str) and _is_scalar(v) for k, v in data.items()):
        parsed.error = "answer_values_not_scalar"
        return parsed
    if not data:
        parsed.error = "answer_empty_object"
        return parsed

    parsed.answer = data
    parsed.format_valid = True
    return parsed
