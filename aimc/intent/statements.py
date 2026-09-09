"""Statement-level parser: English clauses -> statement AST.

How it works
------------
The token stream is parsed by a small recursive-descent driver
(`Parser.sequence` / `Parser.statement`) that understands *structure*:
sentence boundaries, "and"/"then" sequencing, indentation, and the compound
forms `if/otherwise`, `while`, `until`, `repeat ... until`, `keep ... while`.

Everything else is table-driven.  `PATTERNS` is an ordered list of
(regex, handler) pairs.  Each regex is matched with `fullmatch` against the
canonical text of the *longest possible prefix* of the current clause, then
progressively shorter prefixes; the first pattern whose handler accepts wins.
Named groups are handed to the handler as `Groups`, which can turn a group
back into an expression (`g.expr("a")`), a condition (`g.cond("c")`) or an
identifier (`g.ident("var")`).  A handler that raises `ParseFail` simply
rejects that prefix and the search continues, so patterns can be written
loosely and let the expression parser decide.

To teach the compiler a new sentence shape, append a `@pattern(...)` handler
below.  Sugar (fizzbuzz, countdowns, ...) is desugared here into the core
statements of `ast_nodes.py`; lowering never sees it.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple, Union

from .ast_nodes import (Assign, Bin, Call, Exit, Expr, For, If, Logic, Num, Print, Read, Repeat,
                        Stmt, Un, Var, While)
from .expressions import ExprParser, ParseFail, parse_condition_prefix, parse_expr_tokens
from .lexer import Token, text_of, tokenize


class IntentError(Exception):
    """The specification could not be understood."""


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------
STATEMENT_STARTERS = {
    "print", "output", "display", "show", "say", "write", "echo", "emit", "list", "log",
    "set", "let", "define", "declare", "store", "save", "put", "assign", "initialize", "initialise",
    "increase", "increment", "decrease", "decrement", "add", "subtract", "multiply", "divide",
    "double", "halve", "square", "negate", "triple", "read", "ask", "get", "input", "exit", "return",
    "quit", "halt", "stop", "if", "unless", "while", "until", "for", "repeat", "do", "loop", "count",
    "compute", "calculate", "evaluate", "sum", "reverse", "keep", "continue", "else", "otherwise",
    "then", "and", "or", "make", "create", "play", "check", "determine",
}
RESERVED = STATEMENT_STARTERS | {
    "each", "every", "from", "to", "not", "is", "are", "times", "the", "an", "of", "by",
    "with", "in", "into", "it", "its", "their", "whether", "number", "numbers", "integer",
    "integers", "value", "values", "plus", "minus", "product", "difference", "quotient",
    "remainder", "mod", "modulo", "squared", "cubed", "factorial", "even", "odd", "zero",
    "positive", "negative", "prime", "divisible", "multiple", "between", "equal", "equals",
    "greater", "less", "than", "more", "fewer", "at", "least", "most", "stdin", "user", "up",
    "down", "step", "steps", "inclusive", "exclusive", "line", "newline", "space", "nothing",
    "half", "twice", "code", "status", "result", "that", "this", "as", "long", "so", "on",
    "line", "them", "max", "min", "abs", "gcd", "pow", "fib", "fibonacci",
}
SOFT_SEPS = {"and", "then", "also", "next", "afterwards", "afterward", "finally", "subsequently"}
BARE_FORBIDDEN = {"if", "unless", "otherwise", "else", "and", "or", "until", "while", "times",
                  "then", "for", "each", "plus", "minus", "divided", "multiplied", "squared", "cubed",
                  "factorial", "modulo", "mod", "of"} | STATEMENT_STARTERS

ID = r"[a-z_][a-z0-9_]*"
KW_GUARD = r"(?!(?:from|to|in|between|starting|beginning|going|and|is|by|down|up)\b)"


def E(name: str) -> str:
    return r"(?P<%s>.+?)" % name


def V(name: str) -> str:
    return r"(?P<%s>%s%s)" % (name, KW_GUARD, ID)


PRINT = r"(?:print|output|display|show|say|write|echo|emit|list|log)(?: out)?"
NOUN = r"(?:number|integer|value|int|whole number)"
NOUNS = r"(?:numbers|integers|values|ints)"
SRC = r"(?: (?:from|via|using|on|through) (?:the )?(?:stdin|input|standard input|user|keyboard|console|terminal|command line))?"
TO = r"(?:to|through|up to|until|and|thru|till)"
INTO = (r"(?: (?:into|to|as|in|called|named|and call it|and name it|and store it in|and store it as|"
        r"and save it in|and save it as|and keep it in|and put it in|and remember it as|and label it) ")

Handler = Callable[["Parser", "Groups"], Union[Stmt, List[Stmt]]]
PATTERNS: List[Tuple["re.Pattern[str]", Handler, bool]] = []


def pattern(rx: str, compound: bool = False):
    def deco(fn: Handler) -> Handler:
        PATTERNS.append((re.compile(rx), fn, compound))
        return fn
    return deco


# ---------------------------------------------------------------------------
# match groups
# ---------------------------------------------------------------------------
class Groups:
    def __init__(self, p: "Parser", m: "re.Match[str]", pos: int, starts: List[int], words: List[str]):
        self.p = p
        self.m = m
        self.pos = pos
        self.starts = starts
        self.words = words

    def has(self, name: str) -> bool:
        try:
            return self.m.group(name) is not None
        except IndexError:
            return False

    def text(self, name: str) -> str:
        return self.m.group(name)

    def tokens(self, name: str) -> Tuple[int, int]:
        cs, ce = self.m.span(name)
        try:
            i = self.starts.index(cs)
        except ValueError:
            raise ParseFail("group %s is not token aligned" % name)
        j = i
        while j < len(self.words) and self.starts[j] + len(self.words[j]) < ce:
            j += 1
        if j >= len(self.words) or self.starts[j] + len(self.words[j]) != ce:
            raise ParseFail("group %s is not token aligned" % name)
        return self.pos + i, self.pos + j + 1

    def expr(self, name: str, mode: str = "expr") -> Expr:
        i, j = self.tokens(name)
        return parse_expr_tokens(self.p.toks, i, j, self.p, mode)

    def cond(self, name: str) -> Expr:
        return self.expr(name, "cond")

    def ident(self, name: str) -> str:
        t = self.text(name)
        if not re.fullmatch(ID, t) or t in RESERVED:
            raise ParseFail("%r cannot be used as a variable name" % t)
        return t

    def opt_expr(self, name: str, default: int) -> Expr:
        return self.expr(name) if self.has(name) else Num(default)


# ---------------------------------------------------------------------------
# parser driver
# ---------------------------------------------------------------------------
class Parser:
    def __init__(self, toks: List[Token]):
        self.toks = toks
        self.pos = 0
        self.known: set = set()
        self.loop_vars: List[str] = []
        self.last_var: Optional[str] = None
        self.read_vars: List[str] = []
        self.stops: frozenset = frozenset()
        self.best_fail: Tuple[int, str] = (-1, "")

    # -- Context protocol for the expression parser ---------------------------
    def resolve(self, word: str) -> Optional[str]:
        if word == "loop":
            return self.loop_vars[-1] if self.loop_vars else self.last_var
        return self.last_var

    def last_two(self) -> Optional[Tuple[str, str]]:
        if len(self.read_vars) >= 2:
            return self.read_vars[-2], self.read_vars[-1]
        return None

    def declare(self, name: str, read: bool = False) -> None:
        self.known.add(name)
        self.last_var = name
        if read:
            self.read_vars.append(name)

    # -- token helpers -----------------------------------------------------------
    def peek(self, k: int = 0) -> Optional[Token]:
        i = self.pos + k
        return self.toks[i] if i < len(self.toks) else None

    def peek_word(self, k: int = 0) -> Optional[str]:
        t = self.peek(k)
        return t.text if t is not None and t.kind == "word" else None

    def is_punct(self, p: str, k: int = 0) -> bool:
        t = self.peek(k)
        return t is not None and t.kind == "punct" and t.text == p

    def words(self, *seq: str) -> bool:
        for k, w in enumerate(seq):
            if self.peek_word(k) not in w.split("|"):
                return False
        return True

    def take(self, *seq: str) -> bool:
        if self.words(*seq):
            self.pos += len(seq)
            return True
        return False

    def clause_end(self, pos: int) -> int:
        i = pos
        while i < len(self.toks):
            t = self.toks[i]
            if t.kind == "nl" or (t.kind == "punct" and t.text in (".", ";")):
                break
            i += 1
        return i

    def next_non_nl(self, pos: int) -> Optional[Token]:
        while pos < len(self.toks) and self.toks[pos].kind == "nl":
            pos += 1
        return self.toks[pos] if pos < len(self.toks) else None

    def clause_text(self, pos: int) -> str:
        return " ".join(t.raw for t in self.toks[pos:self.clause_end(pos)])

    # -- sequencing --------------------------------------------------------------
    def parse_program(self) -> List[Stmt]:
        stmts = self.sequence(False, -1, frozenset())
        if self.pos < len(self.toks):
            raise IntentError("Unexpected %r; %s" % (self.peek().raw, self.hint()))
        return stmts

    def skip_separators(self, soft: bool, base_indent: int) -> bool:
        while True:
            t = self.peek()
            if t is None:
                return False
            if t.kind == "punct" and t.text in (",", ":"):
                self.pos += 1
                continue
            if t.kind == "punct" and t.text in (".", ";"):
                if soft:
                    return False
                self.pos += 1
                continue
            if t.kind == "nl":
                if soft:
                    nxt = self.next_non_nl(self.pos)
                    if nxt is not None and nxt.indent > base_indent:
                        self.pos += 1
                        continue
                    return False
                self.pos += 1
                continue
            if t.kind == "word":
                if t.text in SOFT_SEPS:
                    self.pos += 1
                    continue
                if t.text == "after" and self.peek_word(1) == "that":
                    self.pos += 2
                    continue
                if t.text in ("end", "done"):
                    nxt = self.peek(1)
                    standalone = nxt is None or nxt.kind == "nl" or (nxt.kind == "punct" and nxt.text in (".", ";"))
                    if standalone:
                        if soft:
                            return False
                        self.pos += 1
                        continue
            return True

    def sequence(self, soft: bool, base_indent: int, stops: frozenset) -> List[Stmt]:
        stmts: List[Stmt] = []
        while True:
            if not self.skip_separators(soft, base_indent):
                # a body may not be empty: allow it to start on the next line
                t = self.peek()
                if soft and not stmts and t is not None and t.kind == "nl":
                    self.pos += 1
                    continue
                break
            if self.peek_word() in stops:
                break
            before = self.pos
            saved = self.stops
            self.stops = stops
            try:
                r = self.statement(base_indent, stops)
            finally:
                self.stops = saved
            stmts.extend(r if isinstance(r, list) else [r])
            if self.pos == before:  # pragma: no cover - safety net
                raise IntentError("parser made no progress at %r" % self.clause_text(self.pos))
        return stmts

    def statement(self, base_indent: int, stops: frozenset) -> Union[Stmt, List[Stmt]]:
        w = self.peek_word()
        if w == "if":
            return self.parse_if(stops)
        if w == "unless":
            return self.parse_if(stops, negate=True)
        if w == "while" or (w in ("as", "so") and self.peek_word(1) == "long"):
            return self.parse_while(stops)
        if w == "until":
            return self.parse_while(stops, negate=True)
        if w in ("keep", "continue"):
            return self.parse_keep(stops)
        r = self.match_patterns()
        if r is None:
            if w in ("repeat", "do"):
                return self.parse_do_until(stops)
            raise IntentError("Could not understand %r; %s" % (self.clause_text(self.pos), self.hint()))
        stmt, compound = r
        if compound:
            return stmt
        return self.trailing_modifiers(stmt, stops)

    def hint(self) -> str:
        if self.best_fail[0] >= 0:
            return "problem: " + self.best_fail[1]
        return "expected a statement such as 'print ...', 'set x to ...', 'if ... then ...', 'for each ... from ... to ...'"

    # -- pattern matching ----------------------------------------------------------
    def match_patterns(self):
        pos = self.pos
        end = self.clause_end(pos)
        words = [text_of(t) for t in self.toks[pos:end]]
        starts: List[int] = []
        c = 0
        for w in words:
            starts.append(c)
            c += len(w) + 1
        prefixes = [" ".join(words[:k]) for k in range(len(words) + 1)]
        self.best_fail = (-1, "")
        for rx, handler, compound in PATTERNS:
            for k in range(len(words), 0, -1):
                m = rx.fullmatch(prefixes[k])
                if not m:
                    continue
                g = Groups(self, m, pos, starts, words)
                self.pos = pos + k
                try:
                    r = handler(self, g)
                except ParseFail as e:
                    if k > self.best_fail[0]:
                        self.best_fail = (k, str(e))
                    self.pos = pos
                    continue
                return r, compound
        self.pos = pos
        return None

    # -- compound statements ---------------------------------------------------------
    def parse_cond_here(self) -> Expr:
        end = self.clause_end(self.pos)
        try:
            cond, newpos = parse_condition_prefix(self.toks, self.pos, end, self)
        except ParseFail as e:
            raise IntentError("Could not understand the condition in %r: %s" % (self.clause_text(self.pos), e))
        self.pos = newpos
        return cond

    def skip_then(self) -> None:
        while True:
            if self.is_punct(",") or self.is_punct(":"):
                self.pos += 1
            elif self.take("then") or self.take("do"):
                pass
            else:
                return

    def take_else(self) -> bool:
        save = self.pos
        while True:
            t = self.peek()
            if t is None:
                break
            if t.kind == "nl" or (t.kind == "punct" and t.text in (",", ".", ";")):
                self.pos += 1
                continue
            break
        if self.take("otherwise") or self.take("else") or self.take("elif") or self.take("or", "else") \
                or self.take("if", "not") or self.take("in", "all", "other", "cases"):
            while self.is_punct(",") or self.is_punct(":") or self.take("then"):
                if self.is_punct(",") or self.is_punct(":"):
                    self.pos += 1
            return True
        self.pos = save
        return False

    def parse_if(self, stops: frozenset, negate: bool = False) -> If:
        base_indent = self.peek().indent
        self.pos += 1
        cond = self.parse_cond_here()
        if negate:
            cond = Un("NOT", cond)
        self.skip_then()
        then = self.sequence(True, base_indent, stops | {"otherwise", "else", "elif"})
        if not then:
            raise IntentError("'if' has no body in %r" % self.clause_text(self.pos))
        otherwise: List[Stmt] = []
        if self.take_else():
            if self.peek_word() == "if":
                otherwise = [self.parse_if(stops)]
            else:
                otherwise = self.sequence(True, base_indent, stops)
        return If(cond, then, otherwise)

    def parse_while(self, stops: frozenset, negate: bool = False) -> While:
        base_indent = self.peek().indent
        if not (self.take("while") or self.take("until") or self.take("as", "long", "as")
                or self.take("so", "long", "as")):
            raise IntentError("expected a loop keyword")
        cond = self.parse_cond_here()
        if negate:
            cond = Un("NOT", cond)
        self.skip_then()
        body = self.sequence(True, base_indent, stops)
        if not body:
            raise IntentError("loop has no body in %r" % self.clause_text(self.pos))
        return While(cond, body)

    def parse_do_until(self, stops: frozenset) -> While:
        base_indent = self.peek().indent
        self.pos += 1  # repeat / do
        self.take("the", "following") or self.take("this")
        self.skip_then()
        body = self.sequence(True, base_indent, stops | {"until", "while", "as"})
        if not body:
            raise IntentError("'repeat' has no body in %r" % self.clause_text(self.pos))
        negate = self.take("until")
        if not negate and not (self.take("while") or self.take("as", "long", "as")):
            raise IntentError("expected 'until' or 'while' after the repeated statements")
        cond = self.parse_cond_here()
        if negate:
            cond = Un("NOT", cond)
        return While(cond, body, do_first=True)

    def parse_keep(self, stops: frozenset) -> While:
        base_indent = self.peek().indent
        self.pos += 1  # keep / continue
        self.take("on") or self.take("doing") or self.take("to")
        body = self.sequence(True, base_indent, stops | {"until", "while", "as"})
        if not body:
            raise IntentError("'keep' has no body in %r" % self.clause_text(self.pos))
        negate = self.take("until")
        if not negate and not (self.take("while") or self.take("as", "long", "as")):
            raise IntentError("expected 'until' or 'while' after 'keep ...'")
        cond = self.parse_cond_here()
        if negate:
            cond = Un("NOT", cond)
        return While(cond, body)

    def trailing_modifiers(self, stmt, stops: frozenset):
        while True:
            w = self.peek_word()
            t = self.peek()
            if w in ("if", "unless"):
                base_indent = t.indent
                self.pos += 1
                cond = self.parse_cond_here()
                if w == "unless":
                    cond = Un("NOT", cond)
                otherwise: List[Stmt] = []
                if self.take_else():
                    otherwise = self.sequence(True, base_indent, stops)
                stmt = If(cond, _as_list(stmt), otherwise)
            elif self.times_ahead():
                count = self.take_times()
                stmt = Repeat(count, _as_list(stmt))
            else:
                return stmt

    def times_ahead(self) -> bool:
        t = self.peek()
        if t is None or self.peek_word(1) != "times":
            return False
        return (t.kind == "num" and not t.ordinal) or (t.kind == "word" and t.text in self.known)

    def take_times(self) -> Expr:
        t = self.peek()
        self.pos += 2
        return Num(t.value) if t.kind == "num" else Var(t.text)

    def body(self, base_indent: int, extra_stops: frozenset = frozenset()) -> List[Stmt]:
        return self.sequence(True, base_indent, self.stops | extra_stops)


def _as_list(x) -> List[Stmt]:
    return list(x) if isinstance(x, list) else [x]


# ---------------------------------------------------------------------------
# helpers for handlers
# ---------------------------------------------------------------------------
def _even(e: Expr) -> Expr:
    return Bin("EQ", Bin("REM", e, Num(2)), Num(0))


def _print_expr(e: Expr) -> Print:
    return Print([("expr", e)])


def _print_str(s: str) -> Print:
    return Print([("str", s)])


def _loop(p: Parser, g: Groups, var: str, start: Expr, end: Expr, step: Expr, down: bool,
          body_fn, parity: Optional[str] = None) -> For:
    """Run `body_fn()` with `var` in scope as the loop variable."""
    p.declare(var)
    p.loop_vars.append(var)
    try:
        body = body_fn()
    finally:
        p.loop_vars.pop()
    if parity == "even":
        body = [If(_even(Var(var)), body)]
    elif parity == "odd":
        body = [If(Un("NOT", _even(Var(var))), body)]
    return For(var, start, end, step, down, body)


def parse_items(p: Parser, i: int, j: int) -> List[Tuple[str, object]]:
    items: List[Tuple[str, object]] = []
    ep = ExprParser(p.toks, i, j, p)
    while True:
        t = ep.tok()
        if t is None:
            ep.fail("expected something to print")
        if t.kind == "str":
            items.append(("str", t.text))
            ep.pos += 1
        else:
            items.append(("expr", ep.negation()))
        if ep.pos >= j:
            return items
        if ep.is_punct(","):
            ep.pos += 1
            ep.take("and")
        elif ep.take("and") or ep.take("followed", "by") or ep.take("then") or ep.take("with"):
            pass
        else:
            ep.fail("unexpected %r" % ep.tok().raw)


_ARITH_NEXT = {"plus", "minus", "times", "divided", "multiplied", "squared", "cubed", "factorial", "mod",
               "modulo", "to", "raised", "doubled", "halved", "is", "equals", "increased", "decreased"}


def bare_string(p: Parser, i: int, j: int) -> str:
    parts: List[str] = []
    for k in range(i, j):
        t = p.toks[k]
        if t.kind == "word":
            if t.text in BARE_FORBIDDEN:
                raise ParseFail("%r cannot appear in a bare string" % t.raw)
            parts.append((" " if parts else "") + t.raw)
        elif t.kind == "num":
            parts.append((" " if parts else "") + t.raw)
        elif t.kind == "punct" and t.text in (",", "'", ":"):
            nxt = p.toks[k + 1] if k + 1 < j else None
            if nxt is None or nxt.kind not in ("word", "num") or nxt.text in BARE_FORBIDDEN:
                raise ParseFail("quote the text you want printed")
            parts.append(t.raw)
        else:
            raise ParseFail("quote the text you want printed")
    nxt = p.toks[j] if j < len(p.toks) else None
    if nxt is not None and nxt.kind == "word" and nxt.text == "times" and p.toks[j - 1].kind == "num":
        raise ParseFail("ambiguous 'N times'")
    if not parts:
        raise ParseFail("nothing to print")
    last = p.toks[j - 1]
    if last.kind == "word" and (last.text in ("the", "of", "value", "values", "result", "to", "by", "with")
                                or (len(parts) > 1 and last.text in ("a", "an"))):
        raise ParseFail("incomplete phrase")
    if nxt is not None and ((nxt.kind == "word" and nxt.text in _ARITH_NEXT)
                            or (nxt.kind == "punct" and nxt.text in ("+", "-", "*", "/", "%", "^", "**", "(", "="))):
        raise ParseFail("unknown variable %r" % p.toks[i].raw)
    return "".join(parts)


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------
def _read(p: Parser, g: Groups, *names: str) -> Read:
    name = "it"
    for n in names:
        if g.has(n):
            name = g.ident(n)
            break
    p.declare(name, read=True)
    return Read(name)


@pattern(r"(?:read|scan|input|receive|accept)(?: in)? (?:a |an |one |the |some |any )?" + NOUN
         + r"(?: (?P<var2>(?!(?:from|to|in|into|as|called|named|and|is|by|via|using|on|through)\b)" + ID + r"))?" + SRC
         + INTO + V("var") + r")?" + SRC)
def h_read_one(p, g):
    return _read(p, g, "var", "var2")


@pattern(r"(?:read|input|scan) (?:in )?(?:two|2|a pair of) " + NOUNS + r"?(?:,? (?:called|named) )?(?: " + V("a") + r" and " + V("b") + r")?" + SRC)
def h_read_two(p, g):
    a = g.ident("a") if g.has("a") else "a"
    b = g.ident("b") if g.has("b") else "b"
    p.declare(a, read=True)
    p.declare(b, read=True)
    return [Read(a), Read(b)]


@pattern(r"(?:read|input|scan) (?:in )?(?:three|3) " + NOUNS + r"?(?:,? (?:called|named) )?(?: " + V("a") + r" , " + V("b") + r" and " + V("c") + r"| " + V("a2") + r" " + V("b2") + r" and " + V("c2") + r")?" + SRC)
def h_read_three(p, g):
    names = [g.ident(n) if g.has(n) else d for n, d in (("a", "a"), ("b", "b"), ("c", "c"))]
    if g.has("a2"):
        names = [g.ident("a2"), g.ident("b2"), g.ident("c2")]
    for n in names:
        p.declare(n, read=True)
    return [Read(n) for n in names]


@pattern(r"(?:read|input|scan|get)(?: in)? (?:the )?(?:" + NOUN + r" )?" + V("var") + SRC)
def h_read_var(p, g):
    return _read(p, g, "var")


@pattern(r"(?:ask|prompt|query) (?:the )?(?:user )?(?:for |to enter |to type |to input |to provide )?(?:a |an )?" + NOUN
         + INTO + V("var") + r")?")
def h_ask(p, g):
    return _read(p, g, "var")


@pattern(r"(?:get|take|obtain|fetch|grab|collect) (?:a |an )?" + NOUN + SRC + INTO + V("var") + r")?")
def h_get(p, g):
    return _read(p, g, "var")


@pattern(r"(?:let|set|initialize|initialise) " + V("var") + r" (?:be|to|=|equal to|equal) (?:a |an |the )?(?:" + NOUN + r" )?"
         r"(?:read|taken|obtained|input|entered|typed|received|provided)(?: in)? (?:from|by|via|on) (?:the )?(?:stdin|input|user|standard input|keyboard|console)")
def h_let_read(p, g):
    return _read(p, g, "var")


@pattern(V("var") + r" (?:=|:=|is) (?:the )?(?:user |user's |next |first )?input(?: " + NOUN + r")?(?: \( \))?|"
         r"(?:let|set) " + V("var2") + r" (?:be|to|=) (?:the )?(?:user |user's |next |first )?input(?: " + NOUN + r")?|"
         + V("var3") + r" = read(?:_int|int|_number)? \( \)")
def h_var_input(p, g):
    return _read(p, g, "var", "var2", "var3")


# ---------------------------------------------------------------------------
# sugar: whole-program idioms
# ---------------------------------------------------------------------------
@pattern(PRINT + r" (?:all |each of |every one of )?(?:the |all the |all of the )?" + NOUNS
         + r" (?:from|between|starting at) " + E("a") + r" " + TO + r" " + E("b")
         + r"(?: inclusive)?(?:,? (?:one per line|each on its own line|on separate lines|each on a new line|in order))?")
def h_print_range(p, g):
    a, b = g.expr("a"), g.expr("b")
    return _loop(p, g, "it", a, b, Num(1), False, lambda: [_print_expr(Var("it"))])


@pattern(PRINT + r" (?:all |each of )?(?:the )?(?P<par>even|odd) " + NOUNS + r" (?:from|between) " + E("a") + r" " + TO + r" " + E("b") + r"(?: inclusive)?")
def h_print_parity(p, g):
    a, b = g.expr("a"), g.expr("b")
    return _loop(p, g, "it", a, b, Num(1), False, lambda: [_print_expr(Var("it"))], parity=g.text("par"))


@pattern(PRINT + r" (?:the )?(?P<pw>squares|cubes) of (?:all |each of )?(?:the )?" + NOUNS + r" (?:from|between) " + E("a") + r" " + TO + r" " + E("b") + r"(?: inclusive)?")
def h_print_powers(p, g):
    a, b = g.expr("a"), g.expr("b")
    v = Var("it")
    e = Bin("MUL", v, v) if g.text("pw") == "squares" else Bin("MUL", Bin("MUL", v, v), v)
    return _loop(p, g, "it", a, b, Num(1), False, lambda: [_print_expr(e)])


@pattern(PRINT + r" (?:the )?(?:first )?" + E("n") + r" multiples of " + E("k"))
def h_print_n_multiples(p, g):
    n, k = g.expr("n"), g.expr("k")
    return _loop(p, g, "it", Num(1), n, Num(1), False, lambda: [_print_expr(Bin("MUL", k, Var("it")))])


@pattern(PRINT + r" (?:all )?(?:the )?multiples of " + E("k") + r" (?P<rel>up to|below|less than|under|not exceeding|to|through|until) " + E("n"))
def h_print_multiples(p, g):
    k, n = g.expr("k"), g.expr("n")
    if g.text("rel") in ("below", "less than", "under"):
        n = Bin("SUB", n, Num(1))
    return _loop(p, g, "it", k, n, k, False, lambda: [_print_expr(Var("it"))])


def _fib_loop(cond_kind: str, limit: Expr) -> List[Stmt]:
    a, b, t = Var("_fa"), Var("_fb"), Var("_ft")
    step = [_print_expr(a), Assign("_ft", Bin("ADD", a, b)), Assign("_fa", b), Assign("_fb", t)]
    init = [Assign("_fa", Num(0)), Assign("_fb", Num(1))]
    if cond_kind == "count":
        return init + [Repeat(limit, step)]
    op = "LE" if cond_kind == "le" else "LT"
    return init + [While(Bin(op, a, limit), step)]


@pattern(PRINT + r" (?:the )?first " + E("n") + r" (?:fibonacci|fib) (?:numbers|terms|values|elements)|"
         + PRINT + r" (?:the )?first " + E("n2") + r" (?:terms|numbers|values|elements) of (?:the )?fibonacci (?:sequence|series)")
def h_fib_first(p, g):
    n = g.expr("n") if g.has("n") else g.expr("n2")
    for v in ("_fa", "_fb", "_ft"):
        p.known.add(v)
    return _fib_loop("count", n)


@pattern(PRINT + r" (?:the )?fibonacci (?:sequence|series|numbers) (?P<rel>up to|below|less than|under|until|to|through|not exceeding|smaller than) " + E("n"))
def h_fib_upto(p, g):
    n = g.expr("n")
    for v in ("_fa", "_fb", "_ft"):
        p.known.add(v)
    kind = "lt" if g.text("rel") in ("below", "less than", "under", "smaller than") else "le"
    return _fib_loop(kind, n)


@pattern(r"(?:(?:" + PRINT + r"|play|do|run|perform|solve|implement) )?fizz ?buzz (?:(?:from|starting at|for) " + E("a") + r" )?(?:up to|to|until|through|and|for) " + E("b") + r"|"
         r"(?:(?:play|do|run|perform|solve|implement) )?fizz ?buzz(?: (?:from|starting at|for) " + E("a2") + r")?")
def h_fizzbuzz(p, g):
    a = g.expr("a") if g.has("a") else g.opt_expr("a2", 1)
    b = g.opt_expr("b", 100)
    v = Var("it")

    def div(k: int) -> Expr:
        return Bin("EQ", Bin("REM", v, Num(k)), Num(0))
    body = [If(div(15), [_print_str("FizzBuzz")],
               [If(div(3), [_print_str("Fizz")],
                   [If(div(5), [_print_str("Buzz")], [_print_expr(v)])])])]
    return _loop(p, g, "it", a, b, Num(1), False, lambda: body)


@pattern(r"count (?:up )?(?:from|starting at) " + E("a") + r" (?:to|up to|through|until) " + E("b")
         + r"(?: and " + PRINT + r" (?:each|every|the) (?:" + NOUN + r"|one)| " + PRINT + r" each " + NOUN + r"| out loud| aloud)?")
def h_count_up(p, g):
    a, b = g.expr("a"), g.expr("b")
    return _loop(p, g, "it", a, b, Num(1), False, lambda: [_print_expr(Var("it"))])


@pattern(r"(?:count ?down|count (?:down|backwards|backward|downwards)) (?:from|starting at) " + E("a") + r"(?: (?:to|down to|until) " + E("b") + r")?|"
         + PRINT + r" a countdown (?:from|starting at) " + E("a2") + r"(?: (?:to|down to) " + E("b2") + r")?")
def h_count_down(p, g):
    a = g.expr("a") if g.has("a") else g.expr("a2")
    b = g.expr("b") if g.has("b") else (g.expr("b2") if g.has("b2") else Num(1))
    return _loop(p, g, "it", a, b, Num(-1), True, lambda: [_print_expr(Var("it"))])


@pattern(r"(?:" + PRINT + r"|check|determine|tell me|say|report) (?:whether|if)(?: or not)? " + E("x") + r" is (?:even or odd|odd or even|an even or odd number|an odd or even number)")
def h_whether_parity(p, g):
    x = g.expr("x")
    return If(_even(x), [_print_str("even")], [_print_str("odd")])


@pattern(r"(?:" + PRINT + r"|check|determine|tell me|say|report) (?:whether|if)(?: or not)? " + E("x") + r" is (?:a )?prime(?: number)?(?: or not)?")
def h_whether_prime(p, g):
    x = g.expr("x")
    return If(Call("is_prime", [x]), [_print_str("prime")], [_print_str("not prime")])


@pattern(r"(?:" + PRINT + r"|check|determine|tell me|say|report) (?:whether|if)(?: or not)? " + E("c"))
def h_whether(p, g):
    c = g.cond("c")
    return If(c, [_print_str("true")], [_print_str("false")])


@pattern(PRINT + r" (?:the )?(?:multiplication|times) table (?:for|of) " + E("n") + r"(?: (?:up to|to|through) " + E("m") + r")?|"
         + PRINT + r" (?:the )?" + E("n2") + r" times table(?: (?:up to|to|through) " + E("m2") + r")?")
def h_times_table(p, g):
    n = g.expr("n") if g.has("n") else g.expr("n2")
    m = g.expr("m") if g.has("m") else (g.expr("m2") if g.has("m2") else Num(10))
    it = Var("it")
    line = Print([("expr", n), ("str", " x "), ("expr", it), ("str", " = "), ("expr", Bin("MUL", n, it))], sep="")
    return _loop(p, g, "it", Num(1), m, Num(1), False, lambda: [line])


@pattern(PRINT + r" (?:the |each |all the )?digits of " + E("n") + r"(?: (?:one per line|on separate lines|each on its own line|in order|from left to right))?")
def h_digits(p, g):
    n = g.expr("n")
    for v in ("_m", "_p"):
        p.known.add(v)
    m, pw = Var("_m"), Var("_p")
    ten = Num(10)
    return [Assign("_m", Call("abs", [n])), Assign("_p", Num(1)),
            While(Bin("LE", Bin("MUL", pw, ten), m), [Assign("_p", Bin("MUL", pw, ten))]),
            While(Bin("GT", pw, Num(0)), [_print_expr(Bin("REM", Bin("DIV", m, pw), ten)),
                                          Assign("_p", Bin("DIV", pw, ten))])]


@pattern(r"(?:reverse|flip|invert) (?:the )?digits of " + E("n") + r"(?: and " + PRINT + r" (?:it|the result|them|the reversed number|the answer))?")
def h_reverse(p, g):
    n = g.expr("n")
    p.declare("result")
    stmts: List[Stmt] = [Assign("result", Call("reverse_digits", [n]))]
    if "print" in g.m.group(0) or "and" in g.m.group(0):
        stmts.append(_print_expr(Var("result")))
    return stmts


@pattern(PRINT + r" (?:the )?first " + E("n") + r" (?:prime numbers|primes)")
def h_first_primes(p, g):
    n = g.expr("n")
    for v in ("_c", "_i"):
        p.known.add(v)
    c, i = Var("_c"), Var("_i")
    return [Assign("_c", Num(0)), Assign("_i", Num(2)),
            While(Bin("LT", c, n), [If(Call("is_prime", [i]), [_print_expr(i), Assign("_c", Bin("ADD", c, Num(1)))]),
                                    Assign("_i", Bin("ADD", i, Num(1)))])]


@pattern(PRINT + r" (?:all |every one of )?(?:the )?(?:prime numbers|primes) (?P<rel>up to|below|less than|under|not exceeding|to|through|between 1 and|between 2 and|from 1 to|from 2 to|until) " + E("n"))
def h_primes_upto(p, g):
    n = g.expr("n")
    if g.text("rel") in ("below", "less than", "under"):
        n = Bin("SUB", n, Num(1))
    return _loop(p, g, "it", Num(2), n, Num(1), False,
                 lambda: [If(Call("is_prime", [Var("it")]), [_print_expr(Var("it"))])])


@pattern(r"(?:" + PRINT + r"|compute|run|generate|do|show) (?:the )?(?:collatz|hailstone) (?:sequence|series|chain|numbers|steps) (?:starting (?:at|from|with)|for|of|from|beginning at|beginning with) " + E("n"))
def h_collatz(p, g):
    n = g.expr("n")
    p.known.add("_h")
    h = Var("_h")
    return [Assign("_h", n),
            While(Bin("NE", h, Num(1)), [_print_expr(h),
                                         If(_even(h), [Assign("_h", Bin("DIV", h, Num(2)))],
                                            [Assign("_h", Bin("ADD", Bin("MUL", h, Num(3)), Num(1)))])]),
            _print_expr(h)]


@pattern(r"(?:sum|add up|total|sum up|add together|total up) (?:all |all of )?(?:the |of the )?(?:" + NOUNS + r" )?(?:from|between) " + E("a") + r" " + TO + r" " + E("b")
         + r"(?: inclusive)?(?: and " + PRINT + r" (?:it|the result|the sum|the total|the answer|that))?")
def h_sum_range(p, g):
    a, b = g.expr("a"), g.expr("b")
    p.declare("result")
    stmts: List[Stmt] = [Assign("result", Call("sum_range", [a, b, Num(1)]))]
    if " print" in g.m.group(0) or " and " in g.m.group(0):
        stmts.append(_print_expr(Var("result")))
    return stmts


@pattern(r"(?:compute|calculate|evaluate|find|determine|work out|figure out|get|take) (?:the )?(?:value |result )?(?:of )?" + E("e")
         + r"(?: and " + PRINT + r" (?:it|the result|the answer|the value|that|out the result))?(?: and store it in " + V("var") + r"| and call it " + V("var2") + r"| as " + V("var3") + r"| into " + V("var4") + r")?")
def h_compute(p, g):
    e = g.expr("e")
    name = "result"
    for n in ("var", "var2", "var3", "var4"):
        if g.has(n):
            name = g.ident(n)
    p.declare(name)
    stmts: List[Stmt] = [Assign(name, e)]
    if re.search(r" (?:print|output|display|show|say|write|echo)", g.m.group(0)):
        stmts.append(_print_expr(Var(name)))
    return stmts


@pattern(PRINT + r" (?:a |an )?(?:blank|empty|new) ?line|" + PRINT + r" (?:a )?newline|" + PRINT + r" nothing|" + PRINT + r" (?:a |an )?(?:blank|empty) string")
def h_blank_line(p, g):
    return Print([])


# ---------------------------------------------------------------------------
# assignment and updates
# ---------------------------------------------------------------------------
def _assign(p: Parser, g: Groups, var_name: str, expr_name: str) -> Assign:
    name = g.ident(var_name)
    e = g.expr(expr_name)
    p.declare(name)
    return Assign(name, e)


@pattern(r"(?:set|initialize|initialise|init|reset|change|update|make) (?:the )?(?:variable |var |value of |counter )?" + V("var")
         + r" (?:to|=|:=|as|equal to|be|to be|to the value|to the value of|start at|start as) " + E("e"))
def h_set(p, g):
    return _assign(p, g, "var", "e")


@pattern(r"let (?:the )?(?:variable |var )?" + V("var") + r" (?:be|=|:=|equal|equal to|have the value|start at|start as|begin at|begin as|hold) " + E("e"))
def h_let(p, g):
    return _assign(p, g, "var", "e")


@pattern(r"(?:define|declare|create|introduce) (?:a |the )?(?:new )?(?:variable |var |counter |number |integer )?(?:called |named )?" + V("var")
         + r" (?:as|to be|=|:=|with (?:the |an? )?(?:initial )?value(?: of)?|equal to|initialized to|initialised to|set to|starting at|and set it to|which is|that is|and initialize it to|and initialise it to) " + E("e"))
def h_define(p, g):
    return _assign(p, g, "var", "e")


@pattern(r"(?:define|declare|create|introduce) (?:a |the )?(?:new )?(?:variable |var |counter |number |integer )?(?:called |named )?" + V("var"))
def h_declare(p, g):
    name = g.ident("var")
    p.declare(name)
    return Assign(name, Num(0))


@pattern(r"(?:store|save|put|keep|place|remember) (?:the )?(?:value |result )?(?:of )?" + E("e")
         + r" (?:in|into|as|to|under) (?:a |the )?(?:new )?(?:variable )?(?:called |named )?" + V("var"))
def h_store(p, g):
    return _assign(p, g, "var", "e")


@pattern(r"assign (?:the )?(?:value )?" + E("e") + r" to (?:the )?(?:variable )?" + V("var") + r"|assign " + V("var2") + r" (?:the value |= |:= )" + E("e2"))
def h_assign(p, g):
    if g.has("var"):
        return _assign(p, g, "var", "e")
    return _assign(p, g, "var2", "e2")


@pattern(V("var") + r" (?:=|:=|is|equals|becomes|is set to|is equal to|shall be|should be|will be|starts at|starts as|begins at|now equals) " + E("e"))
def h_var_eq(p, g):
    return _assign(p, g, "var", "e")


@pattern(r"(?:increase|increment|raise|bump|grow|boost|advance|step) (?:the )?(?:value of |variable |counter )?" + V("var") + r"(?: by " + E("e") + r")?")
def h_increase(p, g):
    name = g.ident("var")
    if name not in p.known:
        raise ParseFail("unknown variable %r" % name)
    p.declare(name)
    return Assign(name, Bin("ADD", Var(name), g.opt_expr("e", 1)))


@pattern(r"(?:decrease|decrement|reduce|lower|shrink|diminish) (?:the )?(?:value of |variable |counter )?" + V("var") + r"(?: by " + E("e") + r")?")
def h_decrease(p, g):
    name = g.ident("var")
    if name not in p.known:
        raise ParseFail("unknown variable %r" % name)
    p.declare(name)
    return Assign(name, Bin("SUB", Var(name), g.opt_expr("e", 1)))


@pattern(r"add " + E("e") + r" to (?:the )?(?:value of |variable |counter )?" + V("var"))
def h_add_to(p, g):
    name = g.ident("var")
    if name not in p.known:
        raise ParseFail("unknown variable %r" % name)
    e = g.expr("e")
    p.declare(name)
    return Assign(name, Bin("ADD", Var(name), e))


@pattern(r"(?:subtract|take|remove|take away) " + E("e") + r" (?:away )?from (?:the )?(?:value of |variable |counter )?" + V("var"))
def h_sub_from(p, g):
    name = g.ident("var")
    if name not in p.known:
        raise ParseFail("unknown variable %r" % name)
    e = g.expr("e")
    p.declare(name)
    return Assign(name, Bin("SUB", Var(name), e))


@pattern(r"(?P<op>multiply|scale|divide) (?:the )?(?:value of |variable )?" + V("var") + r" by " + E("e"))
def h_mul_div(p, g):
    name = g.ident("var")
    if name not in p.known:
        raise ParseFail("unknown variable %r" % name)
    e = g.expr("e")
    p.declare(name)
    return Assign(name, Bin("DIV" if g.text("op") == "divide" else "MUL", Var(name), e))


@pattern(r"(?P<op>double|halve|square|cube|negate|triple|invert the sign of|flip the sign of|zero|clear) (?:the )?(?:value of |variable )?" + V("var"))
def h_unary_update(p, g):
    name = g.ident("var")
    if name not in p.known:
        raise ParseFail("unknown variable %r" % name)
    v = Var(name)
    op = g.text("op")
    e = {"double": Bin("MUL", v, Num(2)), "halve": Bin("DIV", v, Num(2)), "square": Bin("MUL", v, v),
         "cube": Bin("MUL", Bin("MUL", v, v), v), "triple": Bin("MUL", v, Num(3)), "zero": Num(0),
         "clear": Num(0)}.get(op, Un("NEG", v))
    p.declare(name)
    return Assign(name, e)


@pattern(V("var") + r" (?P<op>\+=|-=|\*=|/=|%=) " + E("e") + r"|" + V("var2") + r" (?P<op2>\+\+|--)")
def h_compound_assign(p, g):
    if g.has("var"):
        name, op, e = g.ident("var"), g.text("op"), g.expr("e")
    else:
        name, op, e = g.ident("var2"), g.text("op2"), Num(1)
    if name not in p.known:
        raise ParseFail("unknown variable %r" % name)
    p.declare(name)
    return Assign(name, Bin({"+=": "ADD", "-=": "SUB", "*=": "MUL", "/=": "DIV", "%=": "REM",
                             "++": "ADD", "--": "SUB"}[op], Var(name), e))


# ---------------------------------------------------------------------------
# exit / return
# ---------------------------------------------------------------------------
EXIT = r"(?:exit|quit|terminate|halt|stop|abort|finish)(?: the program| the process| execution| immediately| now)?"


@pattern(EXIT + r"(?: with)?(?: an?)?(?: (?:exit|return|status|error))?(?: (?:code|status|value))?(?: of)?(?: equal to)? " + E("e"))
def h_exit_code(p, g):
    return Exit(g.expr("e"))


@pattern(EXIT + r"(?: with)?(?: an? )?(?:error|failure)")
def h_exit_error(p, g):
    return Exit(Num(1))


@pattern(EXIT + r"(?: (?:normally|successfully|cleanly|gracefully))?(?: with (?:success|no error))?")
def h_exit_plain(p, g):
    return Exit(Num(0))


@pattern(r"return(?: with)?(?: (?:the )?(?:value|code|status))?(?: of)? " + E("e"))
def h_return(p, g):
    return Exit(g.expr("e"), via_return=True)


@pattern(r"return")
def h_return_plain(p, g):
    return Exit(Num(0), via_return=True)


# ---------------------------------------------------------------------------
# loops (compound: handlers parse their own bodies)
# ---------------------------------------------------------------------------
LOOP_TAIL = (r" (?:(?P<down>down to|downto|down through|down until)|(?P<excl>below|less than|under|before)|to|through|up to|until|and|\.\.|ending at|till) " + E("b")
             + r"(?: inclusive)?(?P<excl2> exclusive)?"
             r"(?: (?:in steps of|stepping by|with step|with a step of|step|by|counting by|incrementing by|in increments of|with increment|with an increment of) " + E("step") + r")?"
             r"(?:,? (?:counting|going) (?P<down2>down|downwards|backwards|backward))?(?:,|:| do|,? do the following|,? do this)?")


def _for_body(p: Parser, g: Groups, var: str, a: Expr, b: Expr, step: Optional[Expr], down: bool,
              exclusive: bool, parity: Optional[str]) -> For:
    base_indent = p.toks[g.pos].indent
    if exclusive:
        b = Bin("ADD", b, Num(1)) if down else Bin("SUB", b, Num(1))
    if step is None:
        step = Num(-1) if down else Num(1)
    elif down and isinstance(step, Num) and step.value > 0:
        step = Num(-step.value)
    elif down and not isinstance(step, Num):
        step = Un("NEG", step)

    def body():
        stmts = p.body(base_indent)
        return stmts if stmts else [_print_expr(Var(var))]
    return _loop(p, g, var, a, b, step, down, body, parity)


@pattern(r"for (?:(?:each|every|all) )?(?:(?P<par>even|odd) )?(?:(?:number|integer|value|element|item|iteration|step|counter|index|int) )?"
         r"(?:(?:called|named) )?(?:" + V("var") + r" )?(?:(?:from|starting at|starting from|=|:=|in|between|beginning at|going from|running from|counting from) )?" + E("a") + LOOP_TAIL,
         compound=True)
def h_for(p, g):
    var = g.ident("var") if g.has("var") else "it"
    a, b = g.expr("a"), g.expr("b")
    step = g.expr("step") if g.has("step") else None
    down = g.has("down") or g.has("down2")
    return _for_body(p, g, var, a, b, step, down, g.has("excl") or g.has("excl2"), g.text("par") if g.has("par") else None)


@pattern(r"for " + V("var") + r" in range \( " + E("a") + r" , " + E("b") + r"(?: , " + E("step") + r")? \)(?:,|:| do)?", compound=True)
def h_for_range(p, g):
    var = g.ident("var")
    step = g.expr("step") if g.has("step") else None
    return _for_body(p, g, var, g.expr("a"), g.expr("b"), step, False, True, None)


@pattern(r"(?:loop|iterate|go|run|walk|step|cycle|count) (?:from|over|through|starting at) " + E("a") + LOOP_TAIL.replace("(?P<", "(?P<l_"), compound=True)
def h_loop_from(p, g):
    var = "it"
    step = g.expr("l_step") if g.has("l_step") else None
    down = g.has("l_down") or g.has("l_down2")
    return _for_body(p, g, var, g.expr("a"), g.expr("l_b"), step, down, g.has("l_excl") or g.has("l_excl2"), None)


@pattern(r"(?:from|starting at|starting from) " + E("a") + LOOP_TAIL.replace("(?P<", "(?P<f_") + r"(?: with (?:each|every) (?:number|value) (?:called|named|as) " + V("var") + r")?", compound=True)
def h_from_to(p, g):
    var = g.ident("var") if g.has("var") else "it"
    step = g.expr("f_step") if g.has("f_step") else None
    down = g.has("f_down") or g.has("f_down2")
    return _for_body(p, g, var, g.expr("a"), g.expr("f_b"), step, down, g.has("f_excl") or g.has("f_excl2"), None)


@pattern(r"(?:repeat|do|loop|iterate|run|execute|perform)(?: (?:the following|this|these steps|the next steps|everything below|the steps below))? " + E("n") + r" times(?: over| in a row)?(?:,|:| do)?|"
         r"(?P<n2>[0-9]+|[a-z_][a-z0-9_]*) times(?: over| in a row)?(?:,|:| do)?", compound=True)
def h_repeat(p, g):
    base_indent = p.toks[g.pos].indent
    if g.has("n"):
        n = g.expr("n")
    else:
        t = g.text("n2")
        if t.isdigit():
            n = Num(int(t))
        elif t in p.known:
            n = Var(t)
        else:
            raise ParseFail("unknown repeat count %r" % t)
    body = p.body(base_indent)
    if not body:
        raise IntentError("'repeat' has no body in %r" % p.clause_text(g.pos))
    return Repeat(n, body)


# ---------------------------------------------------------------------------
# generic print (last: the catch-all)
# ---------------------------------------------------------------------------
_SEP_WORDS = {"space": " ", "spaces": " ", "comma": ",", "commas": ",", "tab": "\t", "tabs": "\t",
              "nothing": "", "empty string": "", "newline": "\n", "newlines": "\n", "a comma and a space": ", ",
              "comma and a space": ", ", "comma and space": ", "}


@pattern(PRINT + r" (?:the (?:value|values|result) of )?" + E("items")
         + r"(?P<nonl>,? (?:without a newline|with no newline|on the same line|without a line break|with no line break|without newline|and no newline|and stay on the same line))?"
         + r"(?:,? (?:separated|delimited|joined) by (?:a |an )?(?P<sep>space|spaces|comma|commas|tab|tabs|nothing|empty string|newline|newlines|comma and a space|comma and space))?"
         + r"(?:,? (?:on (?:one|a single|the same) line|all on one line))?(?:,? (?:each )?(?:one per line|on separate lines|on its own line|on their own lines))?")
def h_print(p, g):
    i, j = g.tokens("items")
    newline = not g.has("nonl")
    sep = _SEP_WORDS[g.text("sep")] if g.has("sep") else " "
    try:
        items = parse_items(p, i, j)
    except ParseFail as e:
        if "value of" in g.m.group(0) or "result of" in g.m.group(0):
            raise
        try:
            items = [("str", bare_string(p, i, j))]
        except ParseFail:
            raise e
    return Print(items, sep, newline)


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def parse_statements(text: str) -> List[Stmt]:
    toks = tokenize(text)
    if not toks:
        raise IntentError("The specification is empty")
    return Parser(toks).parse_program()
