"""Recursive-descent parser for arithmetic expressions and conditions.

Works on the token stream from `lexer.py` and accepts both infix notation
(`3 + 4 * x`, `x % 2 == 0`) and English (`the sum of 3 and 4`, `x is
divisible by 2`).  Precedence, loosest to tightest:

    or  <  and  <  not  <  comparison / "is" predicates
    <  + -  <  * / %  <  ^ (right assoc)  <  postfix (squared, factorial, ...)
    <  unary (-, negative, half of, ...)  <  primary

The parser needs a `Context` for variable knowledge and pronoun resolution:
    ctx.known          - set of variable names assigned so far
    ctx.resolve(word)  - name for "it" / "the number" / "the result"
    ctx.last_two()     - names for "their sum" after "read two numbers"
"""
from __future__ import annotations

from typing import List, Optional

from .ast_nodes import Bin, Call, Expr, Logic, Num, Un, Var
from .lexer import Token


class ParseFail(Exception):
    """Pattern did not fit; callers may backtrack and try something else."""


FUNC_ALIASES = {
    "factorial": "factorial", "fact": "factorial", "fib": "fib", "fibonacci": "fib",
    "gcd": "gcd", "max": "max", "min": "min", "abs": "abs", "pow": "pow", "power": "pow",
    "digits": "num_digits", "num_digits": "num_digits", "numdigits": "num_digits",
    "reverse": "reverse_digits", "is_prime": "is_prime", "isprime": "is_prime",
    "prime": "is_prime", "sum_range": "sum_range",
}

_MAX_WORDS = ("maximum", "max", "largest", "larger", "greater", "greatest", "biggest", "bigger",
              "highest", "higher")
_MIN_WORDS = ("minimum", "min", "smallest", "smaller", "least", "lowest", "lower", "fewest")
_NUM_NOUNS = ("numbers", "integers", "values", "ints")
_PRONOUN_NOUNS = ("number", "value", "result", "answer", "total", "count", "counter", "index",
                  "current", "loop", "input", "sum", "product", "difference", "quotient",
                  "remainder", "maximum", "minimum", "gcd", "average")


def _num(v: int) -> Num:
    return Num(v)


def _even(e: Expr) -> Expr:
    return Bin("EQ", Bin("REM", e, _num(2)), _num(0))


def _odd(e: Expr) -> Expr:
    return Bin("NE", Bin("REM", e, _num(2)), _num(0))


class ExprParser:
    def __init__(self, toks: List[Token], start: int, end: int, ctx):
        self.toks = toks
        self.pos = start
        self.end = end
        self.ctx = ctx

    # -- token helpers ---------------------------------------------------------
    def tok(self, k: int = 0) -> Optional[Token]:
        i = self.pos + k
        return self.toks[i] if i < self.end else None

    def is_word(self, w: str, k: int = 0) -> bool:
        t = self.tok(k)
        return t is not None and t.kind == "word" and t.text == w

    def is_punct(self, p: str, k: int = 0) -> bool:
        t = self.tok(k)
        return t is not None and t.kind == "punct" and t.text == p

    def words(self, *seq: str) -> bool:
        for k, w in enumerate(seq):
            t = self.tok(k)
            if t is None or t.kind != "word" or t.text not in w.split("|"):
                return False
        return True

    def take(self, *seq: str) -> bool:
        if self.words(*seq):
            self.pos += len(seq)
            return True
        return False

    def fail(self, msg: str):
        t = self.tok()
        where = "end of clause" if t is None else "%r" % t.raw
        raise ParseFail("%s (at %s)" % (msg, where))

    def attempt(self, fn):
        save = self.pos
        try:
            return fn()
        except ParseFail:
            self.pos = save
            return None

    # -- conditions ------------------------------------------------------------
    def condition(self) -> Expr:
        return self.disjunction()

    def disjunction(self) -> Expr:
        a = self.conjunction()
        while self.is_word("or"):
            save = self.pos
            self.pos += 1
            if self.is_word("else"):
                self.pos = save
                break
            b = self.attempt(self.conjunction)
            if b is None:
                self.pos = save
                break
            a = Logic("or", a, b)
        return a

    def conjunction(self) -> Expr:
        a = self.negation()
        while self.is_word("and") or self.is_word("but"):
            save = self.pos
            self.pos += 1
            b = self.attempt(self.negation)
            if b is None:
                self.pos = save
                break
            a = Logic("and", a, b)
        return a

    def negation(self) -> Expr:
        if self.take("not") or self.take("it", "is", "not", "the", "case", "that"):
            return Un("NOT", self.negation())
        self.take("either") or self.take("both")
        return self.comparison()

    def comparison(self) -> Expr:
        a = self.expr()
        op = self.cmp_operator()
        if op is not None:
            b = self.expr()
            return Bin(op, a, b)
        if self.is_word("is") or self.is_word("are"):
            self.pos += 1
            return self.predicate(a)
        if self.words("does", "not", "equal") or self.words("do", "not", "equal"):
            self.pos += 3
            return Bin("NE", a, self.expr())
        if self.words("does", "not", "exceed"):
            self.pos += 3
            return Bin("LE", a, self.expr())
        if self.take("divides"):
            return Bin("EQ", Bin("REM", self.expr(), a), _num(0))
        return a

    def cmp_operator(self) -> Optional[str]:
        t = self.tok()
        if t is None:
            return None
        if t.kind == "punct":
            m = {"==": "EQ", "=": "EQ", "!=": "NE", "<": "LT", ">": "GT", "<=": "LE", ">=": "GE"}
            if t.text in m:
                self.pos += 1
                return m[t.text]
            return None
        if self.take("equals") or self.take("equal", "to"):
            return "EQ"
        if self.take("exceeds"):
            return "GT"
        if self.take("differs", "from"):
            return "NE"
        return None

    def predicate(self, a: Expr) -> Expr:
        neg = self.take("not")
        self.take("a") or self.take("an")
        r = self._predicate_body(a)
        return Un("NOT", r) if neg else r

    def _predicate_body(self, a: Expr) -> Expr:
        if self.take("even"):
            self.take("number")
            return _even(a)
        if self.take("odd"):
            self.take("number")
            return _odd(a)
        if self.take("zero"):
            return Bin("EQ", a, _num(0))
        if self.take("nonzero"):
            return Bin("NE", a, _num(0))
        if self.take("positive"):
            return Bin("GT", a, _num(0))
        if self.take("negative"):
            return Bin("LT", a, _num(0))
        if self.take("prime"):
            self.take("number")
            return Call("is_prime", [a])
        if self.take("divisible", "by") or self.take("multiple", "of") or self.take("evenly", "divisible", "by"):
            return Bin("EQ", Bin("REM", a, self.expr()), _num(0))
        if self.take("factor", "of") or self.take("divisor", "of"):
            return Bin("EQ", Bin("REM", self.expr(), a), _num(0))
        if self.take("between"):
            lo = self.expr()
            if not self.take("and"):
                self.fail("expected 'and' in 'between A and B'")
            hi = self.expr()
            self.take("inclusive")
            return Logic("and", Bin("GE", a, lo), Bin("LE", a, hi))
        if self.take("equal", "to") or self.take("the", "same", "as"):
            return Bin("EQ", a, self.expr())
        if self.take("different", "from") or self.take("unequal", "to"):
            return Bin("NE", a, self.expr())
        for seq, op in (
            (("greater", "than", "or", "equal", "to"), "GE"), (("at", "least"), "GE"),
            (("no", "less", "than"), "GE"), (("no", "smaller", "than"), "GE"),
            (("less", "than", "or", "equal", "to"), "LE"), (("at", "most"), "LE"),
            (("no", "more", "than"), "LE"), (("no", "greater", "than"), "LE"),
            (("not", "more", "than"), "LE"),
            (("greater", "than"), "GT"), (("more", "than"), "GT"), (("larger", "than"), "GT"),
            (("bigger", "than"), "GT"), (("higher", "than"), "GT"), (("above",), "GT"),
            (("over",), "GT"),
            (("less", "than"), "LT"), (("smaller", "than"), "LT"), (("fewer", "than"), "LT"),
            (("lower", "than"), "LT"), (("below",), "LT"), (("under",), "LT"),
        ):
            if self.take(*seq):
                return Bin(op, a, self.expr())
        # "x is 5"
        return Bin("EQ", a, self.expr())

    # -- arithmetic ------------------------------------------------------------
    def expr(self) -> Expr:
        return self.additive()

    def additive(self) -> Expr:
        a = self.term()
        while True:
            if self.is_punct("+") or self.is_word("plus"):
                self.pos += 1
                a = Bin("ADD", a, self.term())
            elif self.is_punct("-") or self.is_word("minus"):
                self.pos += 1
                a = Bin("SUB", a, self.term())
            elif self.take("increased", "by") or self.take("added", "to"):
                a = Bin("ADD", a, self.term())
            elif self.take("decreased", "by") or self.take("reduced", "by"):
                a = Bin("SUB", a, self.term())
            else:
                return a

    def term(self) -> Expr:
        a = self.power()
        while True:
            if self.is_punct("*") or self.is_word("times") or self.take("multiplied", "by"):
                if self.is_punct("*") or self.is_word("times"):
                    self.pos += 1
                a = Bin("MUL", a, self.power())
            elif self.is_punct("/") or self.take("divided", "by"):
                if self.is_punct("/"):
                    self.pos += 1
                a = Bin("DIV", a, self.power())
            elif self.is_punct("%") or self.is_word("mod") or self.is_word("modulo"):
                self.pos += 1
                a = Bin("REM", a, self.power())
            else:
                return a

    def power(self) -> Expr:
        a = self.postfix()
        if self.is_punct("^") or self.is_punct("**"):
            self.pos += 1
            return Call("pow", [a, self.power()])
        if (self.take("to", "the", "power", "of") or self.take("to", "the", "power")
                or self.take("raised", "to", "the", "power", "of") or self.take("raised", "to", "the", "power")
                or self.take("raised", "to") or self.take("to", "the")):
            b = self.power()
            self.take("power")
            return Call("pow", [a, b])
        return a

    def postfix(self) -> Expr:
        a = self.unary()
        while True:
            if self.take("squared"):
                a = Bin("MUL", a, a)
            elif self.take("cubed"):
                a = Bin("MUL", Bin("MUL", a, a), a)
            elif self.take("factorial") or self.is_punct("!"):
                if self.is_punct("!"):
                    self.pos += 1
                a = Call("factorial", [a])
            elif self.take("doubled"):
                a = Bin("MUL", a, _num(2))
            elif self.take("tripled"):
                a = Bin("MUL", a, _num(3))
            elif self.take("halved"):
                a = Bin("DIV", a, _num(2))
            elif self.take("negated"):
                a = Un("NEG", a)
            elif self.take("reversed") or self.take("backwards"):
                a = Call("reverse_digits", [a])
            elif self.take("percent"):
                a = Bin("DIV", a, _num(100))
            else:
                return a

    def unary(self) -> Expr:
        if self.is_punct("-") or self.take("negative") or self.take("minus"):
            if self.is_punct("-"):
                self.pos += 1
            return Un("NEG", self.unary())
        if self.is_punct("+"):
            self.pos += 1
            return self.unary()
        if self.take("double"):
            return Bin("MUL", _num(2), self.unary())
        if self.take("triple"):
            return Bin("MUL", _num(3), self.unary())
        if self.take("half", "of") or self.take("half"):
            return Bin("DIV", self.unary(), _num(2))
        if self.take("a", "third", "of"):
            return Bin("DIV", self.unary(), _num(3))
        if self.take("a", "quarter", "of"):
            return Bin("DIV", self.unary(), _num(4))
        return self.primary()

    def primary(self) -> Expr:
        t = self.tok()
        if t is None:
            self.fail("expected a value")
        if t.kind == "num":
            if t.ordinal and self.is_word("fibonacci", 1):
                self.pos += 2
                self.take("number") or self.take("term")
                return Call("fib", [_num(t.value)])
            self.pos += 1
            return _num(t.value)
        if t.kind == "punct" and t.text == "(":
            self.pos += 1
            e = self.condition()
            if not self.is_punct(")"):
                self.fail("expected ')'")
            self.pos += 1
            return e
        if t.kind != "word":
            self.fail("unexpected %r" % t.raw)
        w = t.text
        # function-call syntax: name(args)
        if w in FUNC_ALIASES and self.is_punct("(", 1):
            self.pos += 2
            args: List[Expr] = []
            if not self.is_punct(")"):
                while True:
                    args.append(self.condition())
                    if self.is_punct(","):
                        self.pos += 1
                        continue
                    break
            if not self.is_punct(")"):
                self.fail("expected ')'")
            self.pos += 1
            return self._call(FUNC_ALIASES[w], args)
        if w == "the":
            self.pos += 1
            return self.the_form()
        if w == "its":
            self.pos += 1
            return self.of_form(self._pronoun("it"))
        if w == "their":
            self.pos += 1
            return self.their_form()
        if w in ("it", "that", "this"):
            self.pos += 1
            self.take("number") or self.take("value")
            return self._pronoun("it")
        r = self.attempt(self.noun_form)
        if r is not None:
            return r
        if w in self.ctx.known:
            self.pos += 1
            return Var(w)
        if w.isidentifier():
            self.fail("unknown variable %r" % t.raw)
        self.fail("unexpected %r" % t.raw)

    def _pronoun(self, word: str) -> Expr:
        name = self.ctx.resolve(word)
        if name is None:
            self.fail("nothing for %r to refer to" % word)
        return Var(name)

    def _call(self, fn: str, args: List[Expr]) -> Expr:
        arity = {"factorial": 1, "fib": 1, "gcd": 2, "max": 2, "min": 2, "abs": 1, "pow": 2,
                 "num_digits": 1, "reverse_digits": 1, "is_prime": 1, "sum_range": 3}[fn]
        if fn in ("max", "min") and len(args) > 2:
            e = args[0]
            for x in args[1:]:
                e = Call(fn, [e, x])
            return e
        if len(args) != arity:
            self.fail("%s takes %d argument(s)" % (fn, arity))
        return Call(fn, args)

    def their_form(self) -> Expr:
        pair = self.ctx.last_two()
        if pair is None:
            self.fail("'their' needs two previously read values")
        a, b = Var(pair[0]), Var(pair[1])
        for word, op in (("sum", "ADD"), ("total", "ADD"), ("product", "MUL"),
                         ("difference", "SUB"), ("quotient", "DIV"), ("remainder", "REM")):
            if self.take(word):
                return Bin(op, a, b)
        if any(self.take(x) for x in _MAX_WORDS):
            return Call("max", [a, b])
        if any(self.take(x) for x in _MIN_WORDS):
            return Call("min", [a, b])
        if self.take("gcd") or self.take("greatest", "common", "divisor"):
            return Call("gcd", [a, b])
        if self.take("average") or self.take("mean"):
            return Bin("DIV", Bin("ADD", a, b), _num(2))
        self.fail("unknown 'their ...' form")

    def the_form(self) -> Expr:
        """Handles what follows 'the'."""
        t = self.tok()
        if t is None:
            self.fail("expected a noun after 'the'")
        if t.kind == "num" and t.ordinal and self.is_word("fibonacci", 1):
            self.pos += 2
            self.take("number") or self.take("term")
            return Call("fib", [_num(t.value)])
        r = self.attempt(self.noun_form)
        if r is not None:
            return r
        if t.kind == "word":
            w = t.text
            if self.take("value", "of") or self.take("current", "value", "of"):
                return self.primary()
            if w in self.ctx.known:
                self.pos += 1
                return Var(w)
            if self.take("current", "number") or self.take("current", "value") or self.take("loop", "variable") \
                    or self.take("loop", "counter") or self.take("counter") or self.take("index"):
                return self._pronoun("loop")
            if w in ("number", "value", "result", "answer", "total", "input", "integer"):
                self.pos += 1
                return self._pronoun(w)
        self.fail("unexpected %r after 'the'" % t.raw)

    def of_form(self, x: Expr) -> Expr:
        """'its factorial', 'its square' ... applied to x."""
        if self.take("factorial"):
            return Call("factorial", [x])
        if self.take("square"):
            return Bin("MUL", x, x)
        if self.take("cube"):
            return Bin("MUL", Bin("MUL", x, x), x)
        if self.take("absolute", "value"):
            return Call("abs", [x])
        if self.take("number", "of", "digits") or self.take("digit", "count") or self.take("length"):
            return Call("num_digits", [x])
        if self.take("reverse") or self.take("reversal"):
            return Call("reverse_digits", [x])
        if self.take("double"):
            return Bin("MUL", x, _num(2))
        if self.take("half"):
            return Bin("DIV", x, _num(2))
        if self.take("negation") or self.take("negative") or self.take("opposite"):
            return Un("NEG", x)
        if self.take("successor"):
            return Bin("ADD", x, _num(1))
        if self.take("predecessor"):
            return Bin("SUB", x, _num(1))
        self.fail("unknown 'its ...' form")

    def _pair(self, sep: str = "and") -> List[Expr]:
        a = self.expr()
        if not self.take(sep):
            self.fail("expected '%s'" % sep)
        return [a, self.expr()]

    def _list(self) -> List[Expr]:
        items = [self.expr()]
        while True:
            if self.is_punct(","):
                self.pos += 1
                self.take("and")
            elif not self.take("and"):
                break
            e = self.attempt(self.expr)
            if e is None:
                self.pos -= 1
                break
            items.append(e)
        return items

    def noun_form(self) -> Expr:
        """English prefix forms: 'sum of A and B', 'factorial of N', ..."""
        if self.take("sum", "of") or self.take("total", "of"):
            r = self.attempt(self.range_sum)
            if r is not None:
                return r
            items = self._list()
            if len(items) < 2:
                self.fail("'sum of' needs at least two values")
            e = items[0]
            for x in items[1:]:
                e = Bin("ADD", e, x)
            return e
        if self.take("difference", "of") or self.take("difference", "between"):
            a, b = self._pair()
            return Bin("SUB", a, b)
        if self.take("product", "of"):
            items = self._list()
            if len(items) < 2:
                self.fail("'product of' needs at least two values")
            e = items[0]
            for x in items[1:]:
                e = Bin("MUL", e, x)
            return e
        if self.take("quotient", "of"):
            a, b = self._pair()
            return Bin("DIV", a, b)
        if self.take("remainder", "of") or self.take("remainder", "when") or self.take("remainder", "after"):
            a = self.postfix()
            if self.take("is", "divided", "by") or self.take("divided", "by") or self.take("and"):
                return Bin("REM", a, self.power())
            self.fail("expected 'divided by'")
        if self.take("square", "of"):
            a = self.unary()
            return Bin("MUL", a, a)
        if self.take("cube", "of"):
            a = self.unary()
            return Bin("MUL", Bin("MUL", a, a), a)
        if self.take("factorial", "of"):
            return Call("factorial", [self.unary()])
        if self.take("absolute", "value", "of"):
            return Call("abs", [self.unary()])
        if self.take("greatest", "common", "divisor", "of") or self.take("gcd", "of") \
                or self.take("greatest", "common", "factor", "of"):
            a, b = self._pair()
            return Call("gcd", [a, b])
        for w in _MAX_WORDS:
            if self.take(w, "of") or self.take(w, "among") or self.take(w, "between"):
                return self._call("max", self._list())
        for w in _MIN_WORDS:
            if self.take(w, "of") or self.take(w, "among") or self.take(w, "between"):
                return self._call("min", self._list())
        if self.take("number", "of", "digits", "in") or self.take("number", "of", "digits", "of") \
                or self.take("digit", "count", "of") or self.take("length", "of"):
            return Call("num_digits", [self.unary()])
        if self.take("reverse", "of") or self.take("reversal", "of"):
            self.take("the", "digits", "of")
            return Call("reverse_digits", [self.unary()])
        if self.take("negation", "of") or self.take("opposite", "of") or self.take("negative", "of"):
            return Un("NEG", self.unary())
        if self.take("fibonacci", "number") or self.take("fibonacci", "of") or self.take("fibonacci", "number", "at", "position"):
            return Call("fib", [self.unary()])
        if self.take("power", "of"):
            self.fail("'power of' is not a value")
        if self.take("successor", "of"):
            return Bin("ADD", self.unary(), _num(1))
        if self.take("predecessor", "of"):
            return Bin("SUB", self.unary(), _num(1))
        self.fail("no noun form")

    def range_sum(self) -> Expr:
        """'the (even|odd)? numbers from A to B' or 'the first N (even|odd)? numbers' after 'sum of'."""
        self.take("all")
        self.take("the")
        t = self.tok()
        if t is not None and t.kind == "num" and t.ordinal and t.value == 1:
            self.pos += 1
            n = self.expr()
            parity = None
            if self.take("even"):
                parity = "even"
            elif self.take("odd"):
                parity = "odd"
            if not any(self.take(x) for x in _NUM_NOUNS + ("natural", "positive")):
                self.fail("expected 'numbers'")
            self.take("numbers") or self.take("integers")
            if parity == "even":
                return Call("sum_range", [_num(2), Bin("MUL", n, _num(2)), _num(2)])
            if parity == "odd":
                return Call("sum_range", [_num(1), Bin("SUB", Bin("MUL", n, _num(2)), _num(1)), _num(2)])
            return Call("sum_range", [_num(1), n, _num(1)])
        parity = None
        if self.take("even"):
            parity = "even"
        elif self.take("odd"):
            parity = "odd"
        if not any(self.take(x) for x in _NUM_NOUNS):
            self.fail("expected 'numbers'")
        if not (self.take("from") or self.take("between")):
            self.fail("expected 'from'")
        a = self.expr()
        if not (self.take("to") or self.take("and") or self.take("through") or self.take("up", "to")):
            self.fail("expected 'to'")
        b = self.expr()
        self.take("inclusive")
        if parity is None:
            return Call("sum_range", [a, b, _num(1)])
        # first value >= a with the requested parity
        amod = Bin("REM", Bin("ADD", Bin("REM", a, _num(2)), _num(2)), _num(2))  # 0/1 even for negatives
        if parity == "even":
            start = Bin("ADD", a, amod)
        else:
            start = Bin("ADD", a, Bin("SUB", _num(1), amod))
        return Call("sum_range", [start, b, _num(2)])


def parse_expr_tokens(toks: List[Token], start: int, end: int, ctx, mode: str = "expr") -> Expr:
    """Parse toks[start:end] completely as an expression (mode 'expr'), a
    condition ('cond') or a comparison ('cmp'); raise ParseFail otherwise."""
    if start >= end:
        raise ParseFail("expected a value")
    p = ExprParser(toks, start, end, ctx)
    e = {"expr": p.expr, "cond": p.condition, "cmp": p.comparison}[mode]()
    if p.pos != end:
        p.fail("unexpected")
    return e


def parse_condition_prefix(toks: List[Token], start: int, end: int, ctx):
    """Greedy: parse the longest condition starting at `start`; return (expr, next_pos)."""
    p = ExprParser(toks, start, end, ctx)
    e = p.condition()
    return e, p.pos
