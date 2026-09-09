"""Lower the statement AST into IR.

Every named variable gets one slot in `main`, zero-initialised in a prologue
block so the verifier's definite-assignment check never trips on variables
first assigned inside a branch.  Expression results always land in fresh
slots; conditions are lowered to branches with short-circuit semantics.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..ir import BinOp, FunctionBuilder, Intrinsic, IRError, Module, ModuleBuilder, UnOp, verify
from ..ir.nodes import Block
from . import stdlib
from .ast_nodes import (Assign, Bin, Call, Exit, Expr, For, If, Logic, Num, Print, Read, Repeat,
                        Stmt, Un, Var, While)


class LowerError(Exception):
    pass


_BIN = {name: BinOp[name] for name in BinOp.__members__}
_UN = {name: UnOp[name] for name in UnOp.__members__}


class Lowerer:
    def __init__(self) -> None:
        self.mb = ModuleBuilder()
        self.fb: FunctionBuilder = self.mb.function("main", 0)
        self.vars: Dict[str, int] = {}
        self.body_block: Block = self.fb.new_block()
        self.fb.switch_to(self.body_block)

    # -- entry -----------------------------------------------------------------
    def lower_program(self, stmts: List[Stmt]) -> Module:
        for s in stmts:
            self.stmt(s)
        if not self.fb.terminated:
            self.fb.ret(self.fb.const(0))
        # prologue: zero every named variable, then jump to the body
        self.fb.switch_to(self.fb.blocks[0])
        for name in sorted(self.vars, key=self.vars.get):
            self.fb.const(0, dst=self.vars[name])
        self.fb.jmp(self.body_block)
        module = self.mb.build("main")
        try:
            verify(module)
        except IRError as e:  # pragma: no cover - would be a lowering bug
            raise LowerError("internal error: generated IR failed verification: %s" % e)
        return module

    def slot(self, name: str) -> int:
        if name not in self.vars:
            self.vars[name] = self.fb.slot()
        return self.vars[name]

    # -- expressions -----------------------------------------------------------
    def expr(self, e: Expr) -> int:
        fb = self.fb
        if isinstance(e, Num):
            return fb.const(e.value)
        if isinstance(e, Var):
            if e.name not in self.vars:
                raise LowerError("variable %r is used before it is given a value" % e.name)
            return self.vars[e.name]
        if isinstance(e, Bin):
            a = self.expr(e.a)
            b = self.expr(e.b)
            return fb.bin(_BIN[e.op], a, b)
        if isinstance(e, Un):
            return fb.un(_UN[e.op], self.expr(e.a))
        if isinstance(e, Call):
            args = [self.expr(a) for a in e.args]
            idx = stdlib.ensure(self.mb, e.fn)
            return fb.call(idx, args)
        if isinstance(e, Logic):
            result = fb.slot()
            yes, no, join = fb.new_block(), fb.new_block(), fb.new_block()
            self.cond(e, yes, no)
            fb.switch_to(yes)
            fb.const(1, dst=result)
            fb.jmp(join)
            fb.switch_to(no)
            fb.const(0, dst=result)
            fb.jmp(join)
            fb.switch_to(join)
            return result
        raise LowerError("cannot lower expression %r" % (e,))

    def cond(self, e: Expr, then: Block, otherwise: Block) -> None:
        fb = self.fb
        if isinstance(e, Logic):
            mid = fb.new_block()
            if e.op == "and":
                self.cond(e.a, mid, otherwise)
                fb.switch_to(mid)
                self.cond(e.b, then, otherwise)
            else:
                self.cond(e.a, then, mid)
                fb.switch_to(mid)
                self.cond(e.b, then, otherwise)
            return
        if isinstance(e, Un) and e.op == "NOT":
            self.cond(e.a, otherwise, then)
            return
        fb.br(self.expr(e), then, otherwise)

    # -- statements ------------------------------------------------------------
    def block(self, stmts: List[Stmt]) -> None:
        for s in stmts:
            self.stmt(s)

    def stmt(self, s: Stmt) -> None:
        fb = self.fb
        if isinstance(s, Print):
            items = list(s.items)
            for i, (kind, val) in enumerate(items):
                if i > 0 and s.sep:
                    fb.write_str(s.sep.encode("utf-8"))
                if kind == "str":
                    text = val
                    if s.newline and i == len(items) - 1:
                        text += "\n"
                    if text:
                        fb.write_str(text.encode("utf-8"))
                else:
                    fb.intr(Intrinsic.WRITE_INT, [self.expr(val)])
            last_is_str = bool(items) and items[-1][0] == "str"
            if s.newline and not last_is_str:
                fb.intr(Intrinsic.WRITE_CHAR, [fb.const(10)])
        elif isinstance(s, Assign):
            value = self.expr(s.expr)
            fb.mov(value, dst=self.slot(s.name))
        elif isinstance(s, Read):
            fb.intr(Intrinsic.READ_INT, dst=self.slot(s.name))
        elif isinstance(s, Exit):
            value = self.expr(s.expr)
            if s.via_return:
                fb.ret(value)
            else:
                fb.intr(Intrinsic.EXIT, [value])
                fb.ret(value)
            fb.switch_to(fb.new_block())  # unreachable continuation
        elif isinstance(s, If):
            then_b, join = fb.new_block(), fb.new_block()
            else_b = fb.new_block() if s.otherwise else join
            self.cond(s.cond, then_b, else_b)
            fb.switch_to(then_b)
            self.block(s.then)
            if not fb.terminated:
                fb.jmp(join)
            if s.otherwise:
                fb.switch_to(else_b)
                self.block(s.otherwise)
                if not fb.terminated:
                    fb.jmp(join)
            fb.switch_to(join)
        elif isinstance(s, While):
            head, body, done = fb.new_block(), fb.new_block(), fb.new_block()
            fb.jmp(body if s.do_first else head)
            fb.switch_to(head)
            self.cond(s.cond, body, done)
            fb.switch_to(body)
            self.block(s.body)
            if not fb.terminated:
                fb.jmp(head)
            fb.switch_to(done)
        elif isinstance(s, For):
            var = self.slot(s.var)
            fb.mov(self.expr(s.start), dst=var)
            limit = fb.mov(self.expr(s.end))
            step = fb.mov(self.expr(s.step))
            head, body, done = fb.new_block(), fb.new_block(), fb.new_block()
            fb.jmp(head)
            fb.switch_to(head)
            c = fb.bin(BinOp.GE if s.down else BinOp.LE, var, limit)
            fb.br(c, body, done)
            fb.switch_to(body)
            self.block(s.body)
            if not fb.terminated:
                fb.bin(BinOp.ADD, var, step, dst=var)
                fb.jmp(head)
            fb.switch_to(done)
        elif isinstance(s, Repeat):
            counter = fb.mov(self.expr(s.count))
            zero, one = fb.const(0), fb.const(1)
            head, body, done = fb.new_block(), fb.new_block(), fb.new_block()
            fb.jmp(head)
            fb.switch_to(head)
            c = fb.bin(BinOp.GT, counter, zero)
            fb.br(c, body, done)
            fb.switch_to(body)
            self.block(s.body)
            if not fb.terminated:
                fb.bin(BinOp.SUB, counter, one, dst=counter)
                fb.jmp(head)
            fb.switch_to(done)
        else:
            raise LowerError("cannot lower statement %r" % (s,))


def lower(stmts: List[Stmt]) -> Module:
    return Lowerer().lower_program(stmts)
