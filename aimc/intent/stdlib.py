"""Helper routines emitted as IR functions on demand.

Each helper is defined at most once per module and is written directly with
the IR builder, so it goes through the same verifier, interpreter and code
generators as user code.  `factorial` is deliberately recursive to exercise
the call path of every backend.
"""
from __future__ import annotations

from typing import Callable, Dict

from ..ir import BinOp, FunctionBuilder, ModuleBuilder, UnOp

ARITY = {"factorial": 1, "pow": 2, "gcd": 2, "fib": 1, "abs": 1, "is_prime": 1,
         "num_digits": 1, "reverse_digits": 1, "sum_range": 3, "max": 2, "min": 2}


def ensure(mb: ModuleBuilder, name: str) -> int:
    """Return the function index of helper `name`, defining it if necessary."""
    if name not in ARITY:
        raise KeyError(name)
    if mb.has_function(name):
        return mb.declare(name, ARITY[name])
    idx = mb.declare(name, ARITY[name])
    _BUILDERS[name](mb)
    return idx


def _factorial(mb: ModuleBuilder) -> None:
    f = mb.function("factorial", 1)
    n = 0
    one = f.const(1)
    le = f.bin(BinOp.LE, n, one)
    base, rec = f.new_block(), f.new_block()
    f.br(le, base, rec)
    f.switch_to(base)
    f.ret(one)
    f.switch_to(rec)
    nm1 = f.bin(BinOp.SUB, n, one)
    sub = f.call(mb.declare("factorial", 1), [nm1])
    f.ret(f.bin(BinOp.MUL, n, sub))
    mb.define(f)


def _pow(mb: ModuleBuilder) -> None:
    f = mb.function("pow", 2)
    base, exp = 0, 1
    result = f.const(1)
    e = f.mov(exp)
    zero = f.const(0)
    one = f.const(1)
    head, body, done = f.new_block(), f.new_block(), f.new_block()
    f.jmp(head)
    f.switch_to(head)
    c = f.bin(BinOp.GT, e, zero)
    f.br(c, body, done)
    f.switch_to(body)
    f.bin(BinOp.MUL, result, base, dst=result)
    f.bin(BinOp.SUB, e, one, dst=e)
    f.jmp(head)
    f.switch_to(done)
    f.ret(result)
    mb.define(f)


def _gcd(mb: ModuleBuilder) -> None:
    f = mb.function("gcd", 2)
    absf = ensure(mb, "abs")
    a = f.call(absf, [0])
    b = f.call(absf, [1])
    zero = f.const(0)
    head, body, done = f.new_block(), f.new_block(), f.new_block()
    f.jmp(head)
    f.switch_to(head)
    c = f.bin(BinOp.NE, b, zero)
    f.br(c, body, done)
    f.switch_to(body)
    t = f.bin(BinOp.REM, a, b)
    f.mov(b, dst=a)
    f.mov(t, dst=b)
    f.jmp(head)
    f.switch_to(done)
    f.ret(a)
    mb.define(f)


def _fib(mb: ModuleBuilder) -> None:
    f = mb.function("fib", 1)
    n = 0
    zero = f.const(0)
    one = f.const(1)
    le0 = f.bin(BinOp.LE, n, zero)
    ret0, go = f.new_block(), f.new_block()
    f.br(le0, ret0, go)
    f.switch_to(ret0)
    f.ret(zero)
    f.switch_to(go)
    a = f.mov(zero)
    b = f.mov(one)
    i = f.mov(one)
    head, body, done = f.new_block(), f.new_block(), f.new_block()
    f.jmp(head)
    f.switch_to(head)
    c = f.bin(BinOp.LT, i, n)
    f.br(c, body, done)
    f.switch_to(body)
    t = f.bin(BinOp.ADD, a, b)
    f.mov(b, dst=a)
    f.mov(t, dst=b)
    f.bin(BinOp.ADD, i, one, dst=i)
    f.jmp(head)
    f.switch_to(done)
    f.ret(b)
    mb.define(f)


def _abs(mb: ModuleBuilder) -> None:
    f = mb.function("abs", 1)
    zero = f.const(0)
    neg = f.bin(BinOp.LT, 0, zero)
    flip, keep = f.new_block(), f.new_block()
    f.br(neg, flip, keep)
    f.switch_to(flip)
    f.ret(f.un(UnOp.NEG, 0))
    f.switch_to(keep)
    f.ret(0)
    mb.define(f)


def _is_prime(mb: ModuleBuilder) -> None:
    f = mb.function("is_prime", 1)
    n = 0
    zero, one, two = f.const(0), f.const(1), f.const(2)
    small = f.bin(BinOp.LT, n, two)
    no, start = f.new_block(), f.new_block()
    f.br(small, no, start)
    f.switch_to(no)
    f.ret(zero)
    f.switch_to(start)
    i = f.mov(two)
    head, body, yes, step = f.new_block(), f.new_block(), f.new_block(), f.new_block()
    f.jmp(head)
    f.switch_to(head)
    sq = f.bin(BinOp.MUL, i, i)
    c = f.bin(BinOp.LE, sq, n)
    f.br(c, body, yes)
    f.switch_to(body)
    r = f.bin(BinOp.REM, n, i)
    d = f.bin(BinOp.EQ, r, zero)
    f.br(d, no, step)
    f.switch_to(step)
    f.bin(BinOp.ADD, i, one, dst=i)
    f.jmp(head)
    f.switch_to(yes)
    f.ret(one)
    mb.define(f)


def _num_digits(mb: ModuleBuilder) -> None:
    f = mb.function("num_digits", 1)
    n = f.call(ensure(mb, "abs"), [0])
    ten = f.const(10)
    one = f.const(1)
    count = f.const(1)
    head, body, done = f.new_block(), f.new_block(), f.new_block()
    f.jmp(head)
    f.switch_to(head)
    c = f.bin(BinOp.GE, n, ten)
    f.br(c, body, done)
    f.switch_to(body)
    f.bin(BinOp.DIV, n, ten, dst=n)
    f.bin(BinOp.ADD, count, one, dst=count)
    f.jmp(head)
    f.switch_to(done)
    f.ret(count)
    mb.define(f)


def _reverse_digits(mb: ModuleBuilder) -> None:
    f = mb.function("reverse_digits", 1)
    zero, ten = f.const(0), f.const(10)
    neg = f.bin(BinOp.LT, 0, zero)
    n = f.call(ensure(mb, "abs"), [0])
    r = f.const(0)
    head, body, done, flip, keep = (f.new_block() for _ in range(5))
    f.jmp(head)
    f.switch_to(head)
    c = f.bin(BinOp.GT, n, zero)
    f.br(c, body, done)
    f.switch_to(body)
    d = f.bin(BinOp.REM, n, ten)
    r10 = f.bin(BinOp.MUL, r, ten)
    f.bin(BinOp.ADD, r10, d, dst=r)
    f.bin(BinOp.DIV, n, ten, dst=n)
    f.jmp(head)
    f.switch_to(done)
    f.br(neg, flip, keep)
    f.switch_to(flip)
    f.ret(f.un(UnOp.NEG, r))
    f.switch_to(keep)
    f.ret(r)
    mb.define(f)


def _sum_range(mb: ModuleBuilder) -> None:
    f = mb.function("sum_range", 3)
    a, b, step = 0, 1, 2
    s = f.const(0)
    i = f.mov(a)
    head, body, done = f.new_block(), f.new_block(), f.new_block()
    f.jmp(head)
    f.switch_to(head)
    c = f.bin(BinOp.LE, i, b)
    f.br(c, body, done)
    f.switch_to(body)
    f.bin(BinOp.ADD, s, i, dst=s)
    f.bin(BinOp.ADD, i, step, dst=i)
    f.jmp(head)
    f.switch_to(done)
    f.ret(s)
    mb.define(f)


def _minmax(name: str, op: BinOp) -> Callable[[ModuleBuilder], None]:
    def build(mb: ModuleBuilder) -> None:
        f = mb.function(name, 2)
        c = f.bin(op, 0, 1)
        first, second = f.new_block(), f.new_block()
        f.br(c, first, second)
        f.switch_to(first)
        f.ret(0)
        f.switch_to(second)
        f.ret(1)
        mb.define(f)
    return build


_BUILDERS: Dict[str, Callable[[ModuleBuilder], None]] = {
    "factorial": _factorial, "pow": _pow, "gcd": _gcd, "fib": _fib, "abs": _abs,
    "is_prime": _is_prime, "num_digits": _num_digits, "reverse_digits": _reverse_digits,
    "sum_range": _sum_range, "max": _minmax("max", BinOp.GE), "min": _minmax("min", BinOp.LE),
}
