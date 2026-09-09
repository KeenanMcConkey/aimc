"""Ergonomic construction of IR modules.

The builder keeps the IR honest: blocks are created through it, slots are
numbered densely, and `finish()` refuses to hand back a function with an
unterminated block.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .nodes import BinOp, Block, Function, Instr, Intrinsic, Module, Op, UnOp


class FunctionBuilder:
    def __init__(self, module: "ModuleBuilder", name: str, nparams: int):
        self.module = module
        self.name = name
        self.nparams = nparams
        self.nslots = nparams
        self.blocks: List[Block] = []
        self.current: Optional[Block] = None
        self.new_block()  # entry block, id 0

    # -- slots and blocks -----------------------------------------------
    def slot(self) -> int:
        s = self.nslots
        self.nslots += 1
        return s

    def new_block(self) -> Block:
        b = Block(len(self.blocks))
        self.blocks.append(b)
        if self.current is None:
            self.current = b
        return b

    def switch_to(self, block: Block) -> None:
        self.current = block

    @property
    def terminated(self) -> bool:
        return self.current is not None and self.current.terminator is not None

    def _emit(self, instr: Instr) -> Instr:
        assert self.current is not None
        if self.current.terminator is not None:
            raise ValueError(f"block {self.current.id} of {self.name} is already terminated")
        self.current.instrs.append(instr)
        return instr

    # -- data instructions ------------------------------------------------
    def const(self, value: int, dst: Optional[int] = None) -> int:
        dst = self.slot() if dst is None else dst
        self._emit(Instr(Op.CONST, dst=dst, imm=int(value)))
        return dst

    def mov(self, src: int, dst: Optional[int] = None) -> int:
        dst = self.slot() if dst is None else dst
        self._emit(Instr(Op.MOV, dst=dst, args=(src,)))
        return dst

    def bin(self, op: BinOp, a: int, b: int, dst: Optional[int] = None) -> int:
        dst = self.slot() if dst is None else dst
        self._emit(Instr(Op.BIN, sub=int(op), dst=dst, args=(a, b)))
        return dst

    def un(self, op: UnOp, a: int, dst: Optional[int] = None) -> int:
        dst = self.slot() if dst is None else dst
        self._emit(Instr(Op.UN, sub=int(op), dst=dst, args=(a,)))
        return dst

    def call(self, fn_index: int, args: Sequence[int], dst: Optional[int] = None,
             want_result: bool = True) -> Optional[int]:
        if want_result and dst is None:
            dst = self.slot()
        self._emit(Instr(Op.CALL, dst=dst, args=tuple(args), imm=fn_index))
        return dst

    def intr(self, which: Intrinsic, args: Sequence[int] = (), imm: int = 0,
             dst: Optional[int] = None) -> Optional[int]:
        if which == Intrinsic.READ_INT and dst is None:
            dst = self.slot()
        self._emit(Instr(Op.INTR, sub=int(which), dst=dst, args=tuple(args), imm=imm))
        return dst

    def write_str(self, data: bytes) -> None:
        self.intr(Intrinsic.WRITE_STR, imm=self.module.intern_string(data))

    # -- terminators ----------------------------------------------------
    def jmp(self, target: Block) -> None:
        self._emit(Instr(Op.JMP, targets=(target.id,)))

    def br(self, cond: int, then: Block, otherwise: Block) -> None:
        self._emit(Instr(Op.BR, args=(cond,), targets=(then.id, otherwise.id)))

    def ret(self, value: Optional[int] = None) -> None:
        self._emit(Instr(Op.RET, args=() if value is None else (value,)))

    def finish(self) -> Function:
        for b in self.blocks:
            if b.terminator is None:
                raise ValueError(f"block {b.id} of {self.name} lacks a terminator")
        return Function(self.name, self.nparams, self.nslots, self.blocks)


class ModuleBuilder:
    def __init__(self) -> None:
        self.module = Module()
        self._pending: Dict[str, FunctionBuilder] = {}
        self._indices: Dict[str, int] = {}

    def intern_string(self, data: bytes) -> int:
        return self.module.intern_string(data)

    def declare(self, name: str, nparams: int) -> int:
        """Reserve a function index so callers can reference it before its body exists."""
        if name in self._indices:
            return self._indices[name]
        idx = len(self.module.functions)
        self.module.functions.append(Function(name, nparams, nparams, []))
        self._indices[name] = idx
        return idx

    def has_function(self, name: str) -> bool:
        return name in self._indices

    def function(self, name: str, nparams: int) -> FunctionBuilder:
        self.declare(name, nparams)
        fb = FunctionBuilder(self, name, nparams)
        self._pending[name] = fb
        return fb

    def define(self, fb: FunctionBuilder) -> int:
        idx = self._indices[fb.name]
        self.module.functions[idx] = fb.finish()
        self._pending.pop(fb.name, None)
        return idx

    def build(self, entry: str = "main") -> Module:
        for name, fb in list(self._pending.items()):
            self.define(fb)
        self.module.entry = self._indices[entry]
        return self.module
