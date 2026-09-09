"""IR -> x86_64 machine code.

Frame model: every IR slot lives at [rbp - 8*(slot+1)].  Each instruction
loads its operands into rax/rcx, computes, and stores the result -- a direct,
verifiable table walk.  Calls use the System V register convention
(rdi, rsi, rdx, rcx, r8, r9 -> rax), which is also what the runtime routines
and in-process loaders expect.

Image layout:
    _start        process entry: aligns the stack, calls aimc_entry, exits
    aimc_entry    C-ABI wrapper around the IR entry function
    fn<i>         one function per IR function
    rt.*          runtime routines (write_str, write_char, write_int, read_int, exit, trap)
    rodata        interned strings and the trap message
"""
from __future__ import annotations

from aimc.ir.nodes import (TRAP_EXIT_CODE, TRAP_MESSAGE, BinOp, Function, Instr, Intrinsic,
                           Module, Op, UnOp)

from ..image import Image
from ..target import Target
from .encoder import CC, R, X86Assembler

ARG_REGS = [R.RDI, R.RSI, R.RDX, R.RCX, R.R8, R.R9]
CMP_CC = {BinOp.EQ: CC.E, BinOp.NE: CC.NE, BinOp.LT: CC.L, BinOp.LE: CC.LE, BinOp.GT: CC.G, BinOp.GE: CC.GE}


class X86_64Backend:
    def __init__(self, target: Target):
        self.target = target
        self.sys = target.syscalls
        self.a = X86Assembler()

    # ------------------------------------------------------------------
    def compile(self, module: Module) -> Image:
        a = self.a
        self.module = module
        start = a.bind("_start")
        a.and_ri(R.RSP, -16)
        a.call("aimc_entry")
        a.mov_rr(R.RDI, R.RAX)
        a.mov_ri32(R.RAX, self.sys["exit"])
        a.syscall()
        a.ud2()

        entry = a.bind("aimc_entry")
        a.push(R.RBP)
        a.mov_rr(R.RBP, R.RSP)
        a.push(R.RBX)
        a.push(R.R12)     # keep 16-byte alignment (2 pushes after rbp)
        a.call(f"fn{module.entry}")
        a.pop(R.R12)
        a.pop(R.RBX)
        a.pop(R.RBP)
        a.ret()

        for i, f in enumerate(module.functions):
            self._function(i, f)
        self._runtime()
        a.align(16)
        rodata = a.pos
        for i, s in enumerate(module.strings):
            a.bind(f"str{i}")
            a.emit(s)
        a.bind("rt.trap_msg")
        a.emit(TRAP_MESSAGE)
        code = a.finish()
        return Image(self.target, code, start, entry, dict(a.labels), rodata)

    # ------------------------------------------------------------------
    @staticmethod
    def _off(slot: int) -> int:
        return -8 * (slot + 1)

    def _load(self, reg: int, slot: int) -> None:
        self.a.mov_rm(reg, R.RBP, self._off(slot))

    def _store(self, slot: int, reg: int) -> None:
        self.a.mov_mr(R.RBP, self._off(slot), reg)

    def _function(self, index: int, f: Function) -> None:
        a = self.a
        a.align(16)
        a.bind(f"fn{index}")
        frame = (8 * f.nslots + 15) & ~15
        a.push(R.RBP)
        a.mov_rr(R.RBP, R.RSP)
        if frame:
            a.sub_ri(R.RSP, frame)
        for p in range(f.nparams):
            self._store(p, ARG_REGS[p])
        for b in f.blocks:
            a.bind(f"fn{index}.b{b.id}")
            for ins in b.instrs:
                self._instr(index, ins)

    def _instr(self, fi: int, ins: Instr) -> None:
        a = self.a
        op = ins.op
        if op == Op.CONST:
            a.mov_ri(R.RAX, ins.imm)
            self._store(ins.dst, R.RAX)
        elif op == Op.MOV:
            self._load(R.RAX, ins.args[0])
            self._store(ins.dst, R.RAX)
        elif op == Op.BIN:
            self._bin(BinOp(ins.sub), ins)
        elif op == Op.UN:
            self._load(R.RAX, ins.args[0])
            u = UnOp(ins.sub)
            if u == UnOp.NEG:
                a.neg_r(R.RAX)
            elif u == UnOp.BNOT:
                a.not_r(R.RAX)
            else:  # logical NOT
                a.test_rr(R.RAX, R.RAX)
                a.setcc(CC.E, R.RAX)
                a.movzx_rr8(R.RAX, R.RAX)
            self._store(ins.dst, R.RAX)
        elif op == Op.CALL:
            for i, s in enumerate(ins.args):
                self._load(ARG_REGS[i], s)
            a.call(f"fn{ins.imm}")
            if ins.dst is not None:
                self._store(ins.dst, R.RAX)
        elif op == Op.INTR:
            self._intr(Intrinsic(ins.sub), ins)
        elif op == Op.JMP:
            a.jmp(f"fn{fi}.b{ins.targets[0]}")
        elif op == Op.BR:
            self._load(R.RAX, ins.args[0])
            a.test_rr(R.RAX, R.RAX)
            a.jcc(CC.NE, f"fn{fi}.b{ins.targets[0]}")
            a.jmp(f"fn{fi}.b{ins.targets[1]}")
        elif op == Op.RET:
            if ins.args:
                self._load(R.RAX, ins.args[0])
            else:
                a.xor_rr32(R.RAX, R.RAX)
            a.leave()
            a.ret()
        else:  # pragma: no cover
            raise ValueError(op)

    def _bin(self, bop: BinOp, ins: Instr) -> None:
        a = self.a
        self._load(R.RAX, ins.args[0])
        self._load(R.RCX, ins.args[1])
        if bop == BinOp.ADD:
            a.add_rr(R.RAX, R.RCX)
        elif bop == BinOp.SUB:
            a.sub_rr(R.RAX, R.RCX)
        elif bop == BinOp.MUL:
            a.imul_rr(R.RAX, R.RCX)
        elif bop == BinOp.AND:
            a.and_rr(R.RAX, R.RCX)
        elif bop == BinOp.OR:
            a.or_rr(R.RAX, R.RCX)
        elif bop == BinOp.XOR:
            a.xor_rr(R.RAX, R.RCX)
        elif bop == BinOp.SHL:
            a.shl_cl(R.RAX)
        elif bop == BinOp.SHR:
            a.sar_cl(R.RAX)
        elif bop in (BinOp.DIV, BinOp.REM):
            normal, done = a.local("div"), a.local("divdone")
            a.test_rr(R.RCX, R.RCX)
            a.jcc(CC.E, "rt.trap_div0")
            a.cmp_ri(R.RCX, -1)
            a.jcc(CC.NE, normal)
            if bop == BinOp.DIV:
                a.neg_r(R.RAX)          # INT64_MIN / -1 wraps instead of faulting
            else:
                a.xor_rr32(R.RAX, R.RAX)
            a.jmp(done)
            a.bind(normal)
            a.cqo()
            a.idiv_r(R.RCX)
            if bop == BinOp.REM:
                a.mov_rr(R.RAX, R.RDX)
            a.bind(done)
        else:
            a.cmp_rr(R.RAX, R.RCX)
            a.setcc(CMP_CC[bop], R.RAX)
            a.movzx_rr8(R.RAX, R.RAX)
        self._store(ins.dst, R.RAX)

    def _intr(self, which: Intrinsic, ins: Instr) -> None:
        a = self.a
        if which == Intrinsic.WRITE_INT:
            self._load(R.RDI, ins.args[0])
            a.call("rt.write_int")
        elif which == Intrinsic.WRITE_STR:
            a.lea_rip(R.RDI, f"str{ins.imm}")
            a.mov_ri(R.RSI, len(self.module.strings[ins.imm]))
            a.call("rt.write_str")
        elif which == Intrinsic.WRITE_CHAR:
            self._load(R.RDI, ins.args[0])
            a.call("rt.write_char")
        elif which == Intrinsic.READ_INT:
            a.call("rt.read_int")
            self._store(ins.dst, R.RAX)
        elif which == Intrinsic.EXIT:
            self._load(R.RDI, ins.args[0])
            a.call("rt.exit")
        else:  # pragma: no cover
            raise ValueError(which)

    # ------------------------------------------------------------------
    def _runtime(self) -> None:
        a = self.a
        sys_ = self.sys
        macos = self.target.os == "macos"

        # rt.write_str(rdi=ptr, rsi=len): loops until every byte is written.
        a.align(16)
        a.bind("rt.write_str")
        loop, done = a.local("ws"), a.local("wsdone")
        a.bind(loop)
        a.test_rr(R.RSI, R.RSI)
        a.jcc(CC.LE, done)
        a.mov_rr(R.RDX, R.RSI)
        a.mov_rr(R.RSI, R.RDI)
        a.mov_rr(R.R8, R.RDI)          # syscall clobbers rcx/r11 only; keep ptr in r8
        a.mov_rr(R.R9, R.RDX)
        a.mov_ri32(R.RDI, 1)
        a.mov_ri32(R.RAX, sys_["write"])
        a.syscall()
        if macos:
            a.jcc(CC.B, done)          # carry set => error
        a.test_rr(R.RAX, R.RAX)
        a.jcc(CC.LE, done)
        a.mov_rr(R.RDI, R.R8)
        a.add_rr(R.RDI, R.RAX)
        a.mov_rr(R.RSI, R.R9)
        a.sub_rr(R.RSI, R.RAX)
        a.jmp(loop)
        a.bind(done)
        a.ret()

        # rt.write_char(rdi=byte)
        a.align(16)
        a.bind("rt.write_char")
        a.push(R.RDI)                  # byte now at [rsp]
        a.mov_rr(R.RDI, R.RSP)
        a.mov_ri32(R.RSI, 1)
        a.call("rt.write_str")
        a.pop(R.RCX)
        a.ret()

        # rt.write_int(rdi=value): decimal, sign-aware, via a 32-byte stack buffer.
        a.align(16)
        a.bind("rt.write_int")
        a.push(R.RBP)
        a.mov_rr(R.RBP, R.RSP)
        a.sub_ri(R.RSP, 32)
        a.mov_rr(R.RAX, R.RDI)
        a.mov_rr(R.R8, R.RDI)          # remember sign
        a.mov_rr(R.RSI, R.RBP)         # rsi = one past the end of the buffer
        pos, digits, nosign = a.local("pos"), a.local("dig"), a.local("nosign")
        a.test_rr(R.RAX, R.RAX)
        a.jcc(CC.NS, pos)
        a.neg_r(R.RAX)                 # INT64_MIN stays 2**63 as unsigned: still prints correctly
        a.bind(pos)
        a.mov_ri32(R.RCX, 10)
        a.bind(digits)
        a.xor_rr32(R.RDX, R.RDX)
        a.div_r(R.RCX)                 # unsigned divide: rax = quot, rdx = rem
        a.add_ri(R.RDX, ord("0"))
        a.dec_r(R.RSI)
        a.mov_m8r(R.RSI, 0, R.RDX)
        a.test_rr(R.RAX, R.RAX)
        a.jcc(CC.NE, digits)
        a.test_rr(R.R8, R.R8)
        a.jcc(CC.NS, nosign)
        a.dec_r(R.RSI)
        a.mov_m8i(R.RSI, 0, ord("-"))
        a.bind(nosign)
        a.mov_rr(R.RDI, R.RSI)
        a.mov_rr(R.RSI, R.RBP)
        a.sub_rr(R.RSI, R.RDI)         # length
        a.call("rt.write_str")
        a.leave()
        a.ret()

        # rt.getc() -> rax = next byte from fd 0, or -1 at EOF/error.
        a.align(16)
        a.bind("rt.getc")
        eof = a.local("eof")
        a.sub_ri(R.RSP, 16)
        a.xor_rr32(R.RDI, R.RDI)
        a.mov_rr(R.RSI, R.RSP)
        a.mov_ri32(R.RDX, 1)
        a.mov_ri32(R.RAX, sys_["read"])
        a.syscall()
        if macos:
            a.jcc(CC.B, eof)
        a.test_rr(R.RAX, R.RAX)
        a.jcc(CC.LE, eof)
        a.movzx_rm8(R.RAX, R.RSP, 0)
        a.add_ri(R.RSP, 16)
        a.ret()
        a.bind(eof)
        a.mov_ri(R.RAX, -1)
        a.add_ri(R.RSP, 16)
        a.ret()

        # rt.read_int() -> rax (see nodes.py for the exact scanning rules).
        a.align(16)
        a.bind("rt.read_int")
        a.push(R.RBP)
        a.mov_rr(R.RBP, R.RSP)
        a.push(R.R12)                  # value
        a.push(R.R13)                  # negative flag
        a.push(R.R14)                  # digit count
        a.push(R.R15)                  # alignment pad
        a.xor_rr32(R.R12, R.R12)
        a.xor_rr32(R.R13, R.R13)
        a.xor_rr32(R.R14, R.R14)
        skip, first, dig, done, have, positive = (a.local("skip"), a.local("first"), a.local("dig"),
                                                  a.local("done"), a.local("have"), a.local("posv"))
        a.bind(skip)
        a.call("rt.getc")
        for ch in b" \t\n\r":
            a.cmp_ri(R.RAX, ch)
            a.jcc(CC.E, skip)
        a.cmp_ri(R.RAX, ord("-"))
        a.jcc(CC.NE, first)
        a.mov_ri32(R.R13, 1)
        a.call("rt.getc")
        a.bind(first)
        a.bind(dig)
        a.cmp_ri(R.RAX, ord("0"))
        a.jcc(CC.L, done)
        a.cmp_ri(R.RAX, ord("9"))
        a.jcc(CC.G, done)
        a.sub_ri(R.RAX, ord("0"))
        a.imul_rri(R.R12, R.R12, 10)
        a.add_rr(R.R12, R.RAX)
        a.inc_r(R.R14)
        a.call("rt.getc")
        a.jmp(dig)
        a.bind(done)
        a.test_rr(R.R14, R.R14)
        a.jcc(CC.NE, have)
        a.xor_rr32(R.R12, R.R12)
        a.bind(have)
        a.mov_rr(R.RAX, R.R12)
        a.test_rr(R.R13, R.R13)
        a.jcc(CC.E, positive)
        a.neg_r(R.RAX)
        a.bind(positive)
        a.pop(R.R15)
        a.pop(R.R14)
        a.pop(R.R13)
        a.pop(R.R12)
        a.pop(R.RBP)
        a.ret()

        # rt.exit(rdi=code)
        a.align(16)
        a.bind("rt.exit")
        a.mov_ri32(R.RAX, sys_["exit"])
        a.syscall()
        a.ud2()

        # rt.trap_div0: message to stderr, exit(TRAP_EXIT_CODE)
        a.align(16)
        a.bind("rt.trap_div0")
        a.lea_rip(R.RSI, "rt.trap_msg")
        a.mov_ri32(R.RDX, len(TRAP_MESSAGE))
        a.mov_ri32(R.RDI, 2)
        a.mov_ri32(R.RAX, sys_["write"])
        a.syscall()
        a.mov_ri32(R.RDI, TRAP_EXIT_CODE)
        a.jmp("rt.exit")
