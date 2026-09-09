"""Statement / expression AST produced by the intent parser.

This is a *staging* structure private to the front end: the parser builds it
from English, `lower.py` turns it into IR.  It is deliberately tiny -- every
English construct the parser understands is desugared into these few nodes at
parse time, so lowering never has to know about "fizzbuzz" or "countdowns".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple


class Expr:
    pass


@dataclass
class Num(Expr):
    value: int


@dataclass
class Var(Expr):
    name: str


@dataclass
class Bin(Expr):
    op: str          # ADD SUB MUL DIV REM AND OR XOR SHL SHR EQ NE LT LE GT GE
    a: Expr
    b: Expr


@dataclass
class Un(Expr):
    op: str          # NEG NOT BNOT
    a: Expr


@dataclass
class Call(Expr):
    fn: str          # stdlib function name (see stdlib.py)
    args: List[Expr]


@dataclass
class Logic(Expr):
    op: str          # "and" | "or"  (short-circuit)
    a: Expr
    b: Expr


class Stmt:
    pass


@dataclass
class Print(Stmt):
    items: List[Tuple[str, Any]]      # ("expr", Expr) | ("str", str)
    sep: str = " "
    newline: bool = True


@dataclass
class Assign(Stmt):
    name: str
    expr: Expr


@dataclass
class Read(Stmt):
    name: str


@dataclass
class Exit(Stmt):
    expr: Expr
    via_return: bool = False


@dataclass
class If(Stmt):
    cond: Expr
    then: List[Stmt]
    otherwise: List[Stmt] = field(default_factory=list)


@dataclass
class While(Stmt):
    cond: Expr
    body: List[Stmt]
    do_first: bool = False


@dataclass
class For(Stmt):
    var: str
    start: Expr
    end: Expr
    step: Expr
    down: bool
    body: List[Stmt]


@dataclass
class Repeat(Stmt):
    count: Expr
    body: List[Stmt]


# ---------------------------------------------------------------------------
# pretty printer (used by `explain`)
# ---------------------------------------------------------------------------
_BIN_SYM = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/", "REM": "%", "AND": "&", "OR": "|",
            "XOR": "^", "SHL": "<<", "SHR": ">>", "EQ": "==", "NE": "!=", "LT": "<", "LE": "<=",
            "GT": ">", "GE": ">="}


def expr_str(e: Expr) -> str:
    if isinstance(e, Num):
        return str(e.value)
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Bin):
        return "(%s %s %s)" % (expr_str(e.a), _BIN_SYM[e.op], expr_str(e.b))
    if isinstance(e, Un):
        return {"NEG": "-", "NOT": "not ", "BNOT": "~"}[e.op] + expr_str(e.a)
    if isinstance(e, Call):
        return "%s(%s)" % (e.fn, ", ".join(expr_str(a) for a in e.args))
    if isinstance(e, Logic):
        return "(%s %s %s)" % (expr_str(e.a), e.op, expr_str(e.b))
    return repr(e)


def describe(stmts: List[Stmt], indent: int = 0) -> List[str]:
    pad = "  " * indent
    out: List[str] = []
    for s in stmts:
        if isinstance(s, Print):
            parts = [repr(v) if k == "str" else expr_str(v) for k, v in s.items]
            extra = "" if s.newline else " (no newline)"
            out.append(pad + "print " + (", ".join(parts) if parts else "''") + extra)
        elif isinstance(s, Assign):
            out.append(pad + "%s = %s" % (s.name, expr_str(s.expr)))
        elif isinstance(s, Read):
            out.append(pad + "%s = read_int()" % s.name)
        elif isinstance(s, Exit):
            out.append(pad + ("return " if s.via_return else "exit ") + expr_str(s.expr))
        elif isinstance(s, If):
            out.append(pad + "if " + expr_str(s.cond) + ":")
            out.extend(describe(s.then, indent + 1))
            if s.otherwise:
                out.append(pad + "else:")
                out.extend(describe(s.otherwise, indent + 1))
        elif isinstance(s, While):
            if s.do_first:
                out.append(pad + "do:")
                out.extend(describe(s.body, indent + 1))
                out.append(pad + "while " + expr_str(s.cond))
            else:
                out.append(pad + "while " + expr_str(s.cond) + ":")
                out.extend(describe(s.body, indent + 1))
        elif isinstance(s, For):
            out.append(pad + "for %s from %s %s %s step %s:" % (
                s.var, expr_str(s.start), "down to" if s.down else "to", expr_str(s.end), expr_str(s.step)))
            out.extend(describe(s.body, indent + 1))
        elif isinstance(s, Repeat):
            out.append(pad + "repeat %s times:" % expr_str(s.count))
            out.extend(describe(s.body, indent + 1))
        else:
            out.append(pad + repr(s))
    return out
