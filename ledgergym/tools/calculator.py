"""Optional tools for the innovation layer (Section 7.1): calculator / sandbox.

These are deliberately small, dependency-free helpers. A calculator returns a
JSON result the model can quote; the sandbox interface is stubbed so an external
containerized executor can be dropped in without changing the tool contract.
"""

from __future__ import annotations

import ast
import json
import operator
from typing import Any, Callable

__all__ = ["CalculatorError", "safe_eval", "CalculatorTool", "ToolResult"]

_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorError(ValueError):
    """Raised when an expression is not a safe arithmetic expression."""


def safe_eval(expression: str) -> float:
    """Evaluate a pure arithmetic expression (no names, calls, or attributes)."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"could not parse expression: {expression!r}") from exc

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise CalculatorError(f"unsupported constant: {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.BinOp):
            op = _OPS.get(type(node.op))
            if op is None:
                raise CalculatorError(f"unsupported operator: {type(node.op).__name__}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _OPS.get(type(node.op))
            if op is None:
                raise CalculatorError(f"unsupported unary operator: {type(node.op).__name__}")
            return op(_eval(node.operand))
        raise CalculatorError(f"unsupported expression element: {type(node).__name__}")

    return _eval(tree)


class ToolResult(dict):
    """Dict payload returned by a tool; JSON-serializable."""


class CalculatorTool:
    """Evaluate arithmetic and return a JSON-friendly result."""

    name = "calculator"

    def __call__(self, expression: str) -> ToolResult:
        try:
            value = safe_eval(expression)
        except CalculatorError as exc:
            return ToolResult(ok=False, error=str(exc))
        return ToolResult(ok=True, expression=expression, result=round(value, 6))

    @staticmethod
    def to_json(result: ToolResult) -> str:
        return json.dumps(result)
