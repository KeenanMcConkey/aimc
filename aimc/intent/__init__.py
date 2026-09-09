"""Intent parser: natural-language program specifications -> IR.

    from aimc.intent import parse, explain, IntentError
    module = parse("read a number and print its factorial")

The front end is deterministic and rule based (see statements.py for the
pattern table).  `parse` always returns a *verified* IR module or raises
`IntentError`; it never silently produces a program it did not understand.
"""
from __future__ import annotations

from ..ir import Module
from .ast_nodes import describe
from .lower import LowerError, lower
from .statements import IntentError, parse_statements

__all__ = ["parse", "explain", "IntentError"]


def parse(text: str) -> Module:
    stmts = parse_statements(text)
    if not stmts:
        raise IntentError("The specification contains no statements")
    try:
        return lower(stmts)
    except LowerError as e:
        raise IntentError(str(e))


def explain(text: str) -> str:
    """Pseudo-code rendering of what the parser understood."""
    stmts = parse_statements(text)
    return "\n".join(describe(stmts))
