"""Encoder byte-exactness and differential tests of the backends against the interpreter."""
from __future__ import annotations

import unittest

from aimc.codegen import Target, compile_module
from aimc.codegen.aarch64.encoder import A64Assembler, Cond, X
from aimc.codegen.x86_64.encoder import CC, R, X86Assembler
from aimc.ir import BinOp, Intrinsic, ModuleBuilder, UnOp, TRAP_EXIT_CODE, TRAP_MESSAGE

from tests.support import PipelineCase, runnable


class X86EncoderTests(unittest.TestCase):
    def enc(self, fn):
        a = X86Assembler()
        fn(a)
        return a.finish().hex()

    def test_known_encodings(self):
        self.assertEqual(self.enc(lambda a: a.mov_ri32(R.RAX, 42)), "b82a000000")
        self.assertEqual(self.enc(lambda a: a.ret()), "c3")
        self.assertEqual(self.enc(lambda a: a.mov_ri(R.RAX, -1)), "48c7c0ffffffff")
        self.assertEqual(self.enc(lambda a: a.mov_ri(R.R9, 0x1122334455667788)), "49b98877665544332211")
        self.assertEqual(self.enc(lambda a: a.mov_rr(R.RDI, R.RAX)), "4889c7")
        self.assertEqual(self.enc(lambda a: a.mov_rm(R.RAX, R.RBP, -8)), "488b45f8")
        self.assertEqual(self.enc(lambda a: a.mov_mr(R.RBP, -0x100, R.RCX)), "48898d00ffffff")
        self.assertEqual(self.enc(lambda a: a.mov_rm(R.RAX, R.RSP, 0)), "488b0424")      # SIB for rsp base
        self.assertEqual(self.enc(lambda a: a.mov_rm(R.RAX, R.R12, 8)), "498b442408")    # SIB for r12 base
        self.assertEqual(self.enc(lambda a: a.mov_rm(R.RAX, R.R13, 0)), "498b4500")      # r13 needs disp8
        self.assertEqual(self.enc(lambda a: a.imul_rr(R.RAX, R.RCX)), "480fafc1")
        self.assertEqual(self.enc(lambda a: a.imul_rri(R.R12, R.R12, 10)), "4d6be40a")
        self.assertEqual(self.enc(lambda a: a.idiv_r(R.RCX)), "48f7f9")
        self.assertEqual(self.enc(lambda a: a.cqo()), "4899")
        self.assertEqual(self.enc(lambda a: a.setcc(CC.L, R.RAX)), "400f9cc0")
        self.assertEqual(self.enc(lambda a: a.movzx_rr8(R.RAX, R.RAX)), "480fb6c0")
        self.assertEqual(self.enc(lambda a: a.mov_m8r(R.RSI, 0, R.RDX)), "408816")
        self.assertEqual(self.enc(lambda a: a.and_ri(R.RSP, -16)), "4883e4f0")
        self.assertEqual(self.enc(lambda a: a.syscall()), "0f05")
        self.assertEqual(self.enc(lambda a: a.push(R.R12)), "4154")
        self.assertEqual(self.enc(lambda a: a.shl_cl(R.RAX)), "48d3e0")

    def test_relative_fixups(self):
        a = X86Assembler()
        a.bind("top")
        a.jmp("end")           # e9 rel32
        a.call("top")          # e8 rel32
        a.bind("end")
        a.lea_rip(R.RDI, "top")
        code = a.finish()
        self.assertEqual(code[:5].hex(), "e905000000")          # jump over the 5-byte call
        self.assertEqual(code[5:10].hex(), "e8f6ffffff")        # back to offset 0
        self.assertEqual(code[10:].hex(), "488d3defffffff")     # rip-relative back to 0


class A64EncoderTests(unittest.TestCase):
    def enc(self, fn):
        a = A64Assembler()
        fn(a)
        return a.finish().hex()

    def test_known_encodings(self):
        self.assertEqual(self.enc(lambda a: a.mov_ri(X.X0, 42)), "400580d2")
        self.assertEqual(self.enc(lambda a: a.ret()), "c0035fd6")
        self.assertEqual(self.enc(lambda a: a.mov_ri(X.X0, -1)), "000080 92".replace(" ", ""))
        self.assertEqual(self.enc(lambda a: a.mov_ri(X.X1, 0x10000)), "2100a0d2")          # movz x1, #1, lsl #16
        self.assertEqual(self.enc(lambda a: a.mov_ri(X.X2, 0x123456789)), "22f18cd2 a268a4f2 2200c0f2".replace(" ", ""))
        self.assertEqual(self.enc(lambda a: a.mov_ri(X.X3, -2)), "230080 92".replace(" ", ""))   # movn x3, #1
        self.assertEqual(self.enc(lambda a: a.svc(0x80)), "011000d4")
        self.assertEqual(self.enc(lambda a: a.stp_pre(X.FP, X.LR, X.SP, -16)), "fd7bbfa9")
        self.assertEqual(self.enc(lambda a: a.ldp_post(X.FP, X.LR, X.SP, 16)), "fd7bc1a8")
        self.assertEqual(self.enc(lambda a: a.mov_sp(X.FP, X.SP)), "fd030091")
        self.assertEqual(self.enc(lambda a: a.ldr(X.X0, X.SP, 8)), "e00740f9")
        self.assertEqual(self.enc(lambda a: a.str_(X.X1, X.SP, 16)), "e10b00f9")
        self.assertEqual(self.enc(lambda a: a.sdiv(X.X0, X.X0, X.X1)), "000cc19a")
        self.assertEqual(self.enc(lambda a: a.msub(X.X0, X.X2, X.X1, X.X0)), "408001 9b".replace(" ", ""))
        self.assertEqual(self.enc(lambda a: a.cmp_rr(X.X0, X.X1)), "1f0001eb")
        self.assertEqual(self.enc(lambda a: a.cset(X.X0, Cond.LT)), "e0a79f9a")
        self.assertEqual(self.enc(lambda a: a.strb(X.X13, X.X10, 0)), "4d010039")
        self.assertEqual(self.enc(lambda a: a.sub_sp_reg(X.X9)), "ff6329cb")
        self.assertEqual(self.enc(lambda a: a.ldr_reg(X.X0, X.SP, X.X9)), "e06b69f8")
        self.assertEqual(self.enc(lambda a: a.str_reg(X.X0, X.SP, X.X9)), "e06b29f8")

    def test_branch_fixups(self):
        a = A64Assembler()
        a.bind("top")
        a.b("end")
        a.bl("top")
        a.cbz(X.X0, "top")
        a.b_cond(Cond.EQ, "top")
        a.bind("end")
        a.adr(X.X1, "top")
        code = a.finish()
        words = [code[i:i + 4].hex() for i in range(0, len(code), 4)]
        self.assertEqual(words[0], "04000014")   # b +16
        self.assertEqual(words[1], "ffffff97")   # bl -4
        self.assertEqual(words[2], "c0ffffb4")   # cbz x0, -8
        self.assertEqual(words[3], "a0ffff54")   # b.eq -12
        self.assertEqual(words[4], "81ffff10")   # adr x1, -16


# -- differential backend tests -------------------------------------------------
def nl(f):
    f.intr(Intrinsic.WRITE_CHAR, [f.const(10)])


def print_int(f, s):
    f.intr(Intrinsic.WRITE_INT, [s])
    nl(f)


class BackendDifferentialTests(PipelineCase):
    def test_host_can_run_something(self):
        self.assertTrue(runnable(), "no target is executable on this host; the sandbox is misconfigured")

    def test_arithmetic_matrix(self):
        """Every binop / unop over a grid of interesting operands."""
        vals = [0, 1, -1, 2, -2, 7, -7, 10, 63, 64, 65, 12345, -99999, (1 << 62), -(1 << 62), (1 << 63) - 1, -(1 << 63)]
        mb = ModuleBuilder()
        f = mb.function("main", 0)
        consts = {v: f.const(v) for v in vals}
        for op in BinOp:
            for x in vals:
                for y in vals:
                    if op in (BinOp.DIV, BinOp.REM) and y == 0:
                        continue
                    print_int(f, f.bin(op, consts[x], consts[y]))
        for op in UnOp:
            for x in vals:
                print_int(f, f.un(op, consts[x]))
        f.ret(f.const(0))
        m = mb.build()
        self.assertGreater(m.entry_function.nslots, 4096)  # beyond AArch64's immediate offset range
        ref, _ = self.check_module(m)
        self.assertGreater(len(ref.stdout), 10000)

    def test_recursion_and_calls(self):
        mb = ModuleBuilder()
        fact = mb.declare("fact", 1)
        f = mb.function("fact", 1)
        one = f.const(1)
        le = f.bin(BinOp.LE, 0, one)
        base, rec = f.new_block(), f.new_block()
        f.br(le, base, rec)
        f.switch_to(base)
        f.ret(one)
        f.switch_to(rec)
        nm1 = f.bin(BinOp.SUB, 0, one)
        r = f.call(fact, [nm1])
        f.ret(f.bin(BinOp.MUL, 0, r))
        mb.define(f)
        # six-argument function: returns a - b + c*d - e + f
        six = mb.function("six", 6)
        t = six.bin(BinOp.SUB, 0, 1)
        t = six.bin(BinOp.ADD, t, six.bin(BinOp.MUL, 2, 3))
        t = six.bin(BinOp.SUB, t, 4)
        six.ret(six.bin(BinOp.ADD, t, 5))
        six_idx = mb.define(six)
        m = mb.function("main", 0)
        for n in (0, 1, 5, 10, 20, 21):
            print_int(m, m.call(fact, [m.const(n)]))
        print_int(m, m.call(six_idx, [m.const(v) for v in (100, 1, 6, 7, 3, 4)]))
        m.ret(m.const(0))
        ref, _ = self.check_module(mb.build())
        self.assertEqual(ref.stdout, b"1\n1\n120\n3628800\n2432902008176640000\n-4249290049419214848\n142\n")

    def test_loops_and_many_slots(self):
        """700 slots forces disp32 frame offsets on x86 and a >4 KiB frame on AArch64."""
        mb = ModuleBuilder()
        f = mb.function("main", 0)
        slots = [f.const(i) for i in range(700)]
        acc = f.const(0)
        for s in slots[::37]:
            f.bin(BinOp.ADD, acc, s, dst=acc)
        print_int(f, acc)
        # while i < 10: print i*i ; i += 1
        i = f.const(0)
        ten, one = f.const(10), f.const(1)
        head, body, exit_ = f.new_block(), f.new_block(), f.new_block()
        f.jmp(head)
        f.switch_to(head)
        f.br(f.bin(BinOp.LT, i, ten), body, exit_)
        f.switch_to(body)
        print_int(f, f.bin(BinOp.MUL, i, i))
        f.bin(BinOp.ADD, i, one, dst=i)
        f.jmp(head)
        f.switch_to(exit_)
        f.ret(f.const(0))
        ref, _ = self.check_module(mb.build())
        self.assertEqual(ref.stdout.split(b"\n")[0], str(sum(range(0, 700, 37))).encode())
        self.assertEqual(ref.stdout.split(b"\n")[1:11], [str(i * i).encode() for i in range(10)])

    def test_strings_and_io(self):
        mb = ModuleBuilder()
        f = mb.function("main", 0)
        f.write_str(b"Hello, \xc3\xa9 world!\n")
        f.write_str(b"")  # zero-length write
        a = f.intr(Intrinsic.READ_INT)
        b = f.intr(Intrinsic.READ_INT)
        c = f.intr(Intrinsic.READ_INT)  # EOF -> 0
        for s in (a, b, c):
            print_int(f, s)
        f.write_str(b"x" * 5000 + b"\n")
        f.intr(Intrinsic.EXIT, [f.bin(BinOp.ADD, a, b)])
        f.ret()
        ref, _ = self.check_module(mb.build(), stdin=b"\n\n  -17\r\n25")
        self.assertTrue(ref.stdout.startswith(b"Hello, \xc3\xa9 world!\n-17\n25\n0\nxxxx"))
        self.assertEqual(ref.exit_code, 8)

    def test_trap_division_by_zero(self):
        for op in (BinOp.DIV, BinOp.REM):
            mb = ModuleBuilder()
            f = mb.function("main", 0)
            f.write_str(b"before\n")
            zero = f.const(0)
            print_int(f, f.bin(op, f.const(5), zero))
            f.write_str(b"after\n")
            f.ret()
            ref, _ = self.check_module(mb.build(), expected_stdout=b"before\n", expected_exit=TRAP_EXIT_CODE,
                                       expected_stderr=TRAP_MESSAGE)

    def test_exit_code_from_return(self):
        mb = ModuleBuilder()
        f = mb.function("main", 0)
        f.ret(f.const(-1))
        ref, _ = self.check_module(mb.build(), expected_exit=255)

    def test_all_targets_compile(self):
        mb = ModuleBuilder()
        f = mb.function("main", 0)
        f.write_str(b"hi\n")
        f.ret(f.const(3))
        m = mb.build()
        for name in ("x86_64-linux", "x86_64-macos", "aarch64-linux", "aarch64-macos"):
            img = compile_module(m, Target.parse(name))
            self.assertGreater(len(img.code), 100)
            self.assertIn("rt.read_int", img.symbols)
            self.assertEqual(img.start_offset, 0)


if __name__ == "__main__":
    unittest.main()
