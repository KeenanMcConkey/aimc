"""Reference interpreter: the executable specification of IR semantics.

Every backend is validated against this interpreter.  It deliberately mirrors
the machine-level behaviour (64-bit wrapping, truncating division, byte-wise
input scanning) rather than Python's arbitrary-precision arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .nodes import (TRAP_EXIT_CODE, TRAP_MESSAGE, BinOp, Function, Instr, Intrinsic, Module,
                    Op, UnOp, wrap64)


class Fuel(Exception):
    """Raised when the step budget is exhausted (probably an infinite loop)."""


@dataclass
class ExecResult:
    stdout: bytes
    stderr: bytes
    exit_code: int
    steps: int


class _Exit(Exception):
    def __init__(self, code: int):
        self.code = code


class _Trap(Exception):
    pass


@dataclass
class Interpreter:
    module: Module
    stdin: bytes = b""
    fuel: int = 5_000_000
    out: bytearray = field(default_factory=bytearray)
    err: bytearray = field(default_factory=bytearray)
    steps: int = 0
    _in_pos: int = 0

    def run(self) -> ExecResult:
        try:
            rv = self._call(self.module.entry_function, [])
            code = rv & 0xFF
        except _Exit as e:
            code = e.code & 0xFF
        except _Trap:
            self.err += TRAP_MESSAGE
            code = TRAP_EXIT_CODE
        return ExecResult(bytes(self.out), bytes(self.err), code, self.steps)

    # -- execution ------------------------------------------------------
    def _call(self, f: Function, args: List[int]) -> int:
        slots = [0] * f.nslots
        slots[: f.nparams] = args
        block = f.blocks[0]
        while True:
            for ins in block.instrs:
                self.steps += 1
                if self.steps > self.fuel:
                    raise Fuel(f"step budget of {self.fuel} exhausted")
                op = ins.op
                if op == Op.CONST:
                    slots[ins.dst] = wrap64(ins.imm)
                elif op == Op.MOV:
                    slots[ins.dst] = slots[ins.args[0]]
                elif op == Op.BIN:
                    slots[ins.dst] = binop(BinOp(ins.sub), slots[ins.args[0]], slots[ins.args[1]])
                elif op == Op.UN:
                    slots[ins.dst] = unop(UnOp(ins.sub), slots[ins.args[0]])
                elif op == Op.CALL:
                    rv = self._call(self.module.functions[ins.imm], [slots[a] for a in ins.args])
                    if ins.dst is not None:
                        slots[ins.dst] = rv
                elif op == Op.INTR:
                    rv = self._intrinsic(Intrinsic(ins.sub), [slots[a] for a in ins.args], ins.imm)
                    if ins.dst is not None:
                        slots[ins.dst] = rv
                elif op == Op.JMP:
                    block = f.blocks[ins.targets[0]]
                    break
                elif op == Op.BR:
                    block = f.blocks[ins.targets[0] if slots[ins.args[0]] != 0 else ins.targets[1]]
                    break
                elif op == Op.RET:
                    return slots[ins.args[0]] if ins.args else 0
                else:  # pragma: no cover
                    raise ValueError(f"unknown op {op}")

    def _intrinsic(self, which: Intrinsic, args: List[int], imm: int) -> int:
        if which == Intrinsic.WRITE_INT:
            self.out += str(args[0]).encode()
        elif which == Intrinsic.WRITE_STR:
            self.out += self.module.strings[imm]
        elif which == Intrinsic.WRITE_CHAR:
            self.out.append(args[0] & 0xFF)
        elif which == Intrinsic.READ_INT:
            return self._read_int()
        elif which == Intrinsic.EXIT:
            raise _Exit(args[0])
        return 0

    def _read_int(self) -> int:
        data = self.stdin
        n = len(data)
        i = self._in_pos
        while i < n and data[i] in b" \t\r\n":
            i += 1
        neg = False
        if i < n and data[i] == 0x2D:  # '-'
            neg = True
            i += 1
        value = 0
        digits = 0
        while i < n and 0x30 <= data[i] <= 0x39:
            value = wrap64(value * 10 + (data[i] - 0x30))
            digits += 1
            i += 1
        if digits == 0:
            value = 0
        if i < n:
            i += 1  # consume terminator byte
        self._in_pos = i
        return wrap64(-value) if neg else value


def binop(op: BinOp, a: int, b: int) -> int:
    if op == BinOp.ADD:
        return wrap64(a + b)
    if op == BinOp.SUB:
        return wrap64(a - b)
    if op == BinOp.MUL:
        return wrap64(a * b)
    if op == BinOp.DIV:
        if b == 0:
            raise _Trap()
        q = abs(a) // abs(b)
        return wrap64(-q if (a < 0) != (b < 0) else q)
    if op == BinOp.REM:
        if b == 0:
            raise _Trap()
        r = abs(a) % abs(b)
        return wrap64(-r if a < 0 else r)
    if op == BinOp.AND:
        return wrap64(a & b)
    if op == BinOp.OR:
        return wrap64(a | b)
    if op == BinOp.XOR:
        return wrap64(a ^ b)
    if op == BinOp.SHL:
        return wrap64(a << (b & 63))
    if op == BinOp.SHR:
        return wrap64(a >> (b & 63))
    if op == BinOp.EQ:
        return int(a == b)
    if op == BinOp.NE:
        return int(a != b)
    if op == BinOp.LT:
        return int(a < b)
    if op == BinOp.LE:
        return int(a <= b)
    if op == BinOp.GT:
        return int(a > b)
    if op == BinOp.GE:
        return int(a >= b)
    raise ValueError(op)


def unop(op: UnOp, a: int) -> int:
    if op == UnOp.NEG:
        return wrap64(-a)
    if op == UnOp.NOT:
        return int(a == 0)
    if op == UnOp.BNOT:
        return wrap64(~a)
    raise ValueError(op)
