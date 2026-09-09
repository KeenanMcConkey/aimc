"""IR node definitions.

Design notes -- why the IR looks the way it does:

* Every value is a 64-bit two's-complement integer living in a numbered slot
  local to a function.  There are no named variables, no types beyond i64 and
  no nested expressions: each instruction performs exactly one machine-sized
  operation on slots.  Backends map slots 1:1 onto stack-frame words, so
  lowering is a direct table walk and every backend produces bit-identical
  results.
* Control flow is an explicit graph of basic blocks.  Every block ends with
  exactly one terminator (JMP / BR / RET).  The verifier (verify.py) checks
  the graph and definite assignment; the interpreter (interp.py) is the
  executable specification that machine code must reproduce.
* Instructions are positional records with small integer opcodes, so a module
  round-trips through a compact JSON array form and hashes canonically.  That
  is the shape an automated reasoner (or a fuzzer) wants: uniform, total,
  and trivially diffable -- not something meant to be pleasant to read.

Semantics that every executor must honour:

* Arithmetic wraps modulo 2**64.  DIV truncates toward zero; REM has the sign
  of the dividend.  DIV/REM by zero traps: the program writes TRAP_MESSAGE to
  stderr and exits with TRAP_EXIT_CODE.  INT64_MIN / -1 wraps (no trap).
* SHL/SHR shift counts are masked to 6 bits; SHR is arithmetic.
* Comparisons are signed and produce 0 or 1.  NOT is logical (x == 0),
  BNOT is bitwise.
* WRITE_INT writes the signed decimal text of its argument, WRITE_CHAR writes
  the low byte of its argument, WRITE_STR writes an interned byte string.
  READ_INT skips ASCII whitespace, reads an optional '-' and decimal digits,
  consumes the single byte that terminates the digits and yields 0 when no
  digits are found (e.g. at EOF).
* EXIT terminates with the low 8 bits of its argument; falling off the entry
  function exits with the low 8 bits of its return value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

MASK64 = (1 << 64) - 1
TRAP_EXIT_CODE = 70
TRAP_MESSAGE = b"aimc: trap: division by zero\n"


def wrap64(v: int) -> int:
    """Reduce a Python int to a signed 64-bit value."""
    v &= MASK64
    return v - (1 << 64) if v >> 63 else v


class Op(IntEnum):
    CONST = 1   # dst = imm
    MOV = 2     # dst = args[0]
    BIN = 3     # dst = args[0] <sub:BinOp> args[1]
    UN = 4      # dst = <sub:UnOp> args[0]
    CALL = 5    # dst? = functions[imm](*args)
    INTR = 6    # dst? = <sub:Intrinsic>(*args)   (WRITE_STR: imm = string index)
    JMP = 16    # goto targets[0]
    BR = 17     # if args[0] != 0 goto targets[0] else goto targets[1]
    RET = 18    # return args[0] (or 0 when args is empty)


TERMINATORS = frozenset({Op.JMP, Op.BR, Op.RET})


class BinOp(IntEnum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    REM = 4
    AND = 5
    OR = 6
    XOR = 7
    SHL = 8
    SHR = 9
    EQ = 10
    NE = 11
    LT = 12
    LE = 13
    GT = 14
    GE = 15


COMPARISONS = frozenset({BinOp.EQ, BinOp.NE, BinOp.LT, BinOp.LE, BinOp.GT, BinOp.GE})


class UnOp(IntEnum):
    NEG = 0
    NOT = 1
    BNOT = 2


class Intrinsic(IntEnum):
    WRITE_INT = 0
    WRITE_STR = 1
    WRITE_CHAR = 2
    READ_INT = 3
    EXIT = 4


INTRINSIC_ARITY: Dict[Intrinsic, int] = {
    Intrinsic.WRITE_INT: 1,
    Intrinsic.WRITE_STR: 0,
    Intrinsic.WRITE_CHAR: 1,
    Intrinsic.READ_INT: 0,
    Intrinsic.EXIT: 1,
}
INTRINSIC_RETURNS = frozenset({Intrinsic.READ_INT})


@dataclass(frozen=True)
class Instr:
    op: Op
    sub: int = 0
    dst: Optional[int] = None
    args: Tuple[int, ...] = ()
    imm: int = 0
    targets: Tuple[int, ...] = ()

    @property
    def is_terminator(self) -> bool:
        return self.op in TERMINATORS

    def uses(self) -> Tuple[int, ...]:
        return self.args

    def to_list(self) -> list:
        return [int(self.op), int(self.sub), self.dst, list(self.args), self.imm, list(self.targets)]

    @staticmethod
    def from_list(raw: Sequence) -> "Instr":
        op, sub, dst, args, imm, targets = raw
        return Instr(Op(op), int(sub), None if dst is None else int(dst),
                     tuple(int(a) for a in args), int(imm), tuple(int(t) for t in targets))


@dataclass
class Block:
    id: int
    instrs: List[Instr] = field(default_factory=list)

    @property
    def terminator(self) -> Optional[Instr]:
        if self.instrs and self.instrs[-1].is_terminator:
            return self.instrs[-1]
        return None

    @property
    def successors(self) -> Tuple[int, ...]:
        t = self.terminator
        return t.targets if t else ()


@dataclass
class Function:
    name: str
    nparams: int
    nslots: int
    blocks: List[Block] = field(default_factory=list)

    @property
    def entry(self) -> Block:
        return self.blocks[0]

    def predecessors(self) -> Dict[int, List[int]]:
        preds: Dict[int, List[int]] = {b.id: [] for b in self.blocks}
        for b in self.blocks:
            for s in b.successors:
                preds.setdefault(s, []).append(b.id)
        return preds


@dataclass
class Module:
    functions: List[Function] = field(default_factory=list)
    strings: List[bytes] = field(default_factory=list)
    entry: int = 0

    def function_index(self, name: str) -> int:
        for i, f in enumerate(self.functions):
            if f.name == name:
                return i
        raise KeyError(name)

    def intern_string(self, data: bytes) -> int:
        try:
            return self.strings.index(data)
        except ValueError:
            self.strings.append(data)
            return len(self.strings) - 1

    @property
    def entry_function(self) -> Function:
        return self.functions[self.entry]
