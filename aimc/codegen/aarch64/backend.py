"""IR -> AArch64 machine code.

Frame model: `stp x29, x30, [sp, #-16]!; mov x29, sp; sub sp, sp, #frame`;
IR slot i lives at [sp, #8*i].  Operands are loaded into x0/x1, computed,
and stored back.  Calls follow the AAPCS64 register convention (x0-x7 in,
x0 out).  Generated code touches only x0-x17, so callee-saved registers
(x19-x28) and the platform register x18 are never disturbed -- which is
what lets an in-process loader call `aimc_entry` safely.

Syscalls: Linux uses x8 + `svc #0`; macOS uses x16 + `svc #0x80` and reports
errors through the carry flag.
"""
from __future__ import annotations

from aimc.ir.nodes import (TRAP_EXIT_CODE, TRAP_MESSAGE, BinOp, Function, Instr, Intrinsic,
                           Module, Op, UnOp)

from ..image import AsmError, Image
from ..target import Target
from .encoder import A64Assembler, Cond, X

ARG_REGS = [X.X0, X.X1, X.X2, X.X3, X.X4, X.X5, X.X6, X.X7]
CMP_COND = {BinOp.EQ: Cond.EQ, BinOp.NE: Cond.NE, BinOp.LT: Cond.LT, BinOp.LE: Cond.LE,
            BinOp.GT: Cond.GT, BinOp.GE: Cond.GE}


class AArch64Backend:
    def __init__(self, target: Target):
        self.target = target
        self.sys = target.syscalls
        self.macos = target.os == "macos"
        self.a = A64Assembler()

    def _syscall(self, name: str) -> None:
        a = self.a
        if self.macos:
            a.mov_ri(X.X16, self.sys[name])
            a.svc(0x80)
        else:
            a.mov_ri(X.X8, self.sys[name])
            a.svc(0)

    # ------------------------------------------------------------------
    def compile(self, module: Module) -> Image:
        a = self.a
        self.module = module
        start = a.bind("_start")
        a.bl("aimc_entry")
        self._syscall("exit")          # x0 = return value of aimc_entry
        a.brk()

        entry = a.bind("aimc_entry")
        a.stp_pre(X.FP, X.LR, X.SP, -16)
        a.mov_sp(X.FP, X.SP)
        a.bl(f"fn{module.entry}")
        a.ldp_post(X.FP, X.LR, X.SP, 16)
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
    # Slot offsets up to 32760 use the scaled 12-bit immediate form; beyond
    # that the offset is materialised in x15 and the register-offset form used.
    def _load(self, reg: int, slot: int) -> None:
        off = 8 * slot
        if off <= 32760:
            self.a.ldr(reg, X.SP, off)
        else:
            self.a.mov_ri(X.X15, off)
            self.a.ldr_reg(reg, X.SP, X.X15)

    def _store(self, slot: int, reg: int) -> None:
        off = 8 * slot
        if off <= 32760:
            self.a.str_(reg, X.SP, off)
        else:
            self.a.mov_ri(X.X15, off)
            self.a.str_reg(reg, X.SP, X.X15)

    def _function(self, index: int, f: Function) -> None:
        a = self.a
        a.align(16)
        a.bind(f"fn{index}")
        frame = (8 * f.nslots + 15) & ~15
        a.stp_pre(X.FP, X.LR, X.SP, -16)
        a.mov_sp(X.FP, X.SP)
        if frame:
            if frame < 4096:
                a.sub_ri(X.SP, X.SP, frame)
            else:
                a.mov_ri(X.X9, frame)
                a.sub_sp_reg(X.X9)
        for p in range(f.nparams):
            self._store(p, ARG_REGS[p])
        for b in f.blocks:
            a.bind(f"fn{index}.b{b.id}")
            for ins in b.instrs:
                self._instr(index, ins)

    def _epilogue(self) -> None:
        a = self.a
        a.mov_sp(X.SP, X.FP)
        a.ldp_post(X.FP, X.LR, X.SP, 16)
        a.ret()

    def _instr(self, fi: int, ins: Instr) -> None:
        a = self.a
        op = ins.op
        if op == Op.CONST:
            a.mov_ri(X.X0, ins.imm)
            self._store(ins.dst, X.X0)
        elif op == Op.MOV:
            self._load(X.X0, ins.args[0])
            self._store(ins.dst, X.X0)
        elif op == Op.BIN:
            self._bin(BinOp(ins.sub), ins)
        elif op == Op.UN:
            self._load(X.X0, ins.args[0])
            u = UnOp(ins.sub)
            if u == UnOp.NEG:
                a.neg(X.X0, X.X0)
            elif u == UnOp.BNOT:
                a.mvn(X.X0, X.X0)
            else:
                a.cmp_ri(X.X0, 0)
                a.cset(X.X0, Cond.EQ)
            self._store(ins.dst, X.X0)
        elif op == Op.CALL:
            for i, s in enumerate(ins.args):
                self._load(ARG_REGS[i], s)
            a.bl(f"fn{ins.imm}")
            if ins.dst is not None:
                self._store(ins.dst, X.X0)
        elif op == Op.INTR:
            self._intr(Intrinsic(ins.sub), ins)
        elif op == Op.JMP:
            a.b(f"fn{fi}.b{ins.targets[0]}")
        elif op == Op.BR:
            self._load(X.X0, ins.args[0])
            a.cbnz(X.X0, f"fn{fi}.b{ins.targets[0]}")
            a.b(f"fn{fi}.b{ins.targets[1]}")
        elif op == Op.RET:
            if ins.args:
                self._load(X.X0, ins.args[0])
            else:
                a.mov_ri(X.X0, 0)
            self._epilogue()
        else:  # pragma: no cover
            raise ValueError(op)

    def _bin(self, bop: BinOp, ins: Instr) -> None:
        a = self.a
        self._load(X.X0, ins.args[0])
        self._load(X.X1, ins.args[1])
        if bop == BinOp.ADD:
            a.add_rr(X.X0, X.X0, X.X1)
        elif bop == BinOp.SUB:
            a.sub_rr(X.X0, X.X0, X.X1)
        elif bop == BinOp.MUL:
            a.mul(X.X0, X.X0, X.X1)
        elif bop == BinOp.AND:
            a.and_rr(X.X0, X.X0, X.X1)
        elif bop == BinOp.OR:
            a.orr_rr(X.X0, X.X0, X.X1)
        elif bop == BinOp.XOR:
            a.eor_rr(X.X0, X.X0, X.X1)
        elif bop == BinOp.SHL:
            a.lslv(X.X0, X.X0, X.X1)
        elif bop == BinOp.SHR:
            a.asrv(X.X0, X.X0, X.X1)
        elif bop == BinOp.DIV:
            a.cbz(X.X1, "rt.trap_div0")
            a.sdiv(X.X0, X.X0, X.X1)          # INT64_MIN / -1 wraps in hardware, as specified
        elif bop == BinOp.REM:
            a.cbz(X.X1, "rt.trap_div0")
            a.sdiv(X.X2, X.X0, X.X1)
            a.msub(X.X0, X.X2, X.X1, X.X0)   # x0 - (x0 / x1) * x1
        else:
            a.cmp_rr(X.X0, X.X1)
            a.cset(X.X0, CMP_COND[bop])
        self._store(ins.dst, X.X0)

    def _intr(self, which: Intrinsic, ins: Instr) -> None:
        a = self.a
        if which == Intrinsic.WRITE_INT:
            self._load(X.X0, ins.args[0])
            a.bl("rt.write_int")
        elif which == Intrinsic.WRITE_STR:
            a.adr(X.X0, f"str{ins.imm}")
            a.mov_ri(X.X1, len(self.module.strings[ins.imm]))
            a.bl("rt.write_str")
        elif which == Intrinsic.WRITE_CHAR:
            self._load(X.X0, ins.args[0])
            a.bl("rt.write_char")
        elif which == Intrinsic.READ_INT:
            a.bl("rt.read_int")
            self._store(ins.dst, X.X0)
        elif which == Intrinsic.EXIT:
            self._load(X.X0, ins.args[0])
            a.bl("rt.exit")
        else:  # pragma: no cover
            raise ValueError(which)

    # ------------------------------------------------------------------
    def _runtime(self) -> None:
        a = self.a

        # rt.write_str(x0=ptr, x1=len)
        a.align(16)
        a.bind("rt.write_str")
        loop, done = a.local("ws"), a.local("wsdone")
        a.mov_rr(X.X9, X.X0)
        a.mov_rr(X.X10, X.X1)
        a.bind(loop)
        a.cmp_ri(X.X10, 0)
        a.b_cond(Cond.LE, done)
        a.mov_ri(X.X0, 1)
        a.mov_rr(X.X1, X.X9)
        a.mov_rr(X.X2, X.X10)
        self._syscall("write")
        if self.macos:
            a.b_cond(Cond.CS, done)
        a.cmp_ri(X.X0, 0)
        a.b_cond(Cond.LE, done)
        a.add_rr(X.X9, X.X9, X.X0)
        a.sub_rr(X.X10, X.X10, X.X0)
        a.b(loop)
        a.bind(done)
        a.ret()

        # rt.write_char(x0=byte)
        a.align(16)
        a.bind("rt.write_char")
        a.stp_pre(X.FP, X.LR, X.SP, -32)
        a.strb(X.X0, X.SP, 16)
        a.add_ri(X.X0, X.SP, 16)
        a.mov_ri(X.X1, 1)
        a.bl("rt.write_str")
        a.ldp_post(X.FP, X.LR, X.SP, 32)
        a.ret()

        # rt.write_int(x0=value)
        a.align(16)
        a.bind("rt.write_int")
        pos, digits, nosign = a.local("pos"), a.local("dig"), a.local("nosign")
        a.stp_pre(X.FP, X.LR, X.SP, -48)     # 32-byte digit buffer at [sp+16, sp+48)
        a.mov_rr(X.X9, X.X0)                 # sign
        a.cmp_ri(X.X0, 0)
        a.b_cond(Cond.GE, pos)
        a.neg(X.X0, X.X0)                    # INT64_MIN stays 2**63 unsigned; udiv handles it
        a.bind(pos)
        a.add_ri(X.X10, X.SP, 48)            # end pointer
        a.mov_ri(X.X11, 10)
        a.bind(digits)
        a.udiv(X.X12, X.X0, X.X11)
        a.msub(X.X13, X.X12, X.X11, X.X0)    # remainder
        a.add_ri(X.X13, X.X13, ord("0"))
        a.sub_ri(X.X10, X.X10, 1)
        a.strb(X.X13, X.X10, 0)
        a.mov_rr(X.X0, X.X12)
        a.cbnz(X.X0, digits)
        a.cmp_ri(X.X9, 0)
        a.b_cond(Cond.GE, nosign)
        a.mov_ri(X.X13, ord("-"))
        a.sub_ri(X.X10, X.X10, 1)
        a.strb(X.X13, X.X10, 0)
        a.bind(nosign)
        a.mov_rr(X.X0, X.X10)
        a.add_ri(X.X1, X.SP, 48)
        a.sub_rr(X.X1, X.X1, X.X10)
        a.bl("rt.write_str")
        a.ldp_post(X.FP, X.LR, X.SP, 48)
        a.ret()

        # rt.getc() -> x0 = byte or -1
        a.align(16)
        a.bind("rt.getc")
        eof = a.local("eof")
        a.sub_ri(X.SP, X.SP, 16)
        a.mov_ri(X.X0, 0)
        a.mov_sp(X.X1, X.SP)
        a.mov_ri(X.X2, 1)
        self._syscall("read")
        if self.macos:
            a.b_cond(Cond.CS, eof)
        a.cmp_ri(X.X0, 0)
        a.b_cond(Cond.LE, eof)
        a.ldrb(X.X0, X.SP, 0)
        a.add_ri(X.SP, X.SP, 16)
        a.ret()
        a.bind(eof)
        a.mov_ri(X.X0, -1)
        a.add_ri(X.SP, X.SP, 16)
        a.ret()

        # rt.read_int() -> x0
        a.align(16)
        a.bind("rt.read_int")
        skip, first, dig, done, have, fin = (a.local("skip"), a.local("first"), a.local("dig"),
                                             a.local("done"), a.local("have"), a.local("fin"))
        a.stp_pre(X.FP, X.LR, X.SP, -48)
        a.stp(X.X19, X.X20, X.SP, 16)
        a.str_(X.X21, X.SP, 32)
        a.mov_ri(X.X19, 0)                   # value
        a.mov_ri(X.X20, 0)                   # negative flag
        a.mov_ri(X.X21, 0)                   # digit count
        a.bind(skip)
        a.bl("rt.getc")
        for ch in b" \t\n\r":
            a.cmp_ri(X.X0, ch)
            a.b_cond(Cond.EQ, skip)
        a.cmp_ri(X.X0, ord("-"))
        a.b_cond(Cond.NE, first)
        a.mov_ri(X.X20, 1)
        a.bl("rt.getc")
        a.bind(first)
        a.bind(dig)
        a.cmp_ri(X.X0, ord("0"))
        a.b_cond(Cond.LT, done)
        a.cmp_ri(X.X0, ord("9"))
        a.b_cond(Cond.GT, done)
        a.sub_ri(X.X0, X.X0, ord("0"))
        a.mov_ri(X.X9, 10)
        a.madd(X.X19, X.X19, X.X9, X.X0)
        a.add_ri(X.X21, X.X21, 1)
        a.bl("rt.getc")
        a.b(dig)
        a.bind(done)
        a.cbnz(X.X21, have)
        a.mov_ri(X.X19, 0)
        a.bind(have)
        a.mov_rr(X.X0, X.X19)
        a.cbz(X.X20, fin)
        a.neg(X.X0, X.X0)
        a.bind(fin)
        a.ldr(X.X21, X.SP, 32)
        a.ldp(X.X19, X.X20, X.SP, 16)
        a.ldp_post(X.FP, X.LR, X.SP, 48)
        a.ret()

        # rt.exit(x0=code)
        a.align(16)
        a.bind("rt.exit")
        self._syscall("exit")
        a.brk()

        # rt.trap_div0
        a.align(16)
        a.bind("rt.trap_div0")
        a.adr(X.X1, "rt.trap_msg")
        a.mov_ri(X.X2, len(TRAP_MESSAGE))
        a.mov_ri(X.X0, 2)
        self._syscall("write")
        a.mov_ri(X.X0, TRAP_EXIT_CODE)
        a.b("rt.exit")
