"""IR construction, verification, serialisation and interpreter semantics."""
from __future__ import annotations

import unittest

from aimc.ir import (BinOp, Instr, Interpreter, IRError, Intrinsic, ModuleBuilder, Op, UnOp, digest,
                     disassemble, from_json, to_json, verify, wrap64, TRAP_EXIT_CODE, TRAP_MESSAGE)
from aimc.ir.interp import binop, unop
from aimc.ir.nodes import Block, Function, Module


def simple_module():
    mb = ModuleBuilder()
    f = mb.function("main", 0)
    a = f.const(6)
    b = f.const(7)
    c = f.bin(BinOp.MUL, a, b)
    f.intr(Intrinsic.WRITE_INT, [c])
    f.intr(Intrinsic.WRITE_CHAR, [f.const(10)])
    f.ret(f.const(0))
    return mb.build()


class BuilderTests(unittest.TestCase):
    def test_build_and_run(self):
        m = simple_module()
        verify(m)
        r = Interpreter(m).run()
        self.assertEqual(r.stdout, b"42\n")
        self.assertEqual(r.exit_code, 0)

    def test_unterminated_block_rejected(self):
        mb = ModuleBuilder()
        f = mb.function("main", 0)
        f.const(1)
        with self.assertRaises(ValueError):
            mb.build()

    def test_emit_after_terminator_rejected(self):
        mb = ModuleBuilder()
        f = mb.function("main", 0)
        f.ret()
        with self.assertRaises(ValueError):
            f.const(1)


class VerifierTests(unittest.TestCase):
    def test_use_before_def(self):
        f = Function("main", 0, 2, [Block(0, [Instr(Op.MOV, dst=1, args=(0,)), Instr(Op.RET, args=(1,))])])
        with self.assertRaisesRegex(IRError, "used before definition"):
            verify(Module([f]))

    def test_def_on_all_paths_ok_but_one_path_missing_fails(self):
        # b0: c = const 1 ; br c ? b1 : b2 ; b1: x = const 5 ; jmp b3 ; b2: jmp b3 ; b3: ret x
        blocks = [
            Block(0, [Instr(Op.CONST, dst=0, imm=1), Instr(Op.BR, args=(0,), targets=(1, 2))]),
            Block(1, [Instr(Op.CONST, dst=1, imm=5), Instr(Op.JMP, targets=(3,))]),
            Block(2, [Instr(Op.JMP, targets=(3,))]),
            Block(3, [Instr(Op.RET, args=(1,))]),
        ]
        with self.assertRaisesRegex(IRError, "slot 1 may be used before definition"):
            verify(Module([Function("main", 0, 2, blocks)]))
        blocks[2] = Block(2, [Instr(Op.CONST, dst=1, imm=6), Instr(Op.JMP, targets=(3,))])
        verify(Module([Function("main", 0, 2, blocks)]))  # now defined on both paths

    def test_bad_targets_and_arity(self):
        f = Function("main", 0, 1, [Block(0, [Instr(Op.JMP, targets=(7,))])])
        with self.assertRaisesRegex(IRError, "out of range"):
            verify(Module([f]))
        f = Function("main", 0, 1, [Block(0, [Instr(Op.INTR, sub=int(Intrinsic.WRITE_INT), args=()), Instr(Op.RET)])])
        with self.assertRaisesRegex(IRError, "expected 1 args"):
            verify(Module([f]))

    def test_entry_must_take_no_params(self):
        f = Function("main", 1, 1, [Block(0, [Instr(Op.RET)])])
        with self.assertRaisesRegex(IRError, "0 parameters"):
            verify(Module([f]))

    def test_terminator_in_middle(self):
        f = Function("main", 0, 1, [Block(0, [Instr(Op.RET), Instr(Op.RET)])])
        with self.assertRaisesRegex(IRError, "terminator placement"):
            verify(Module([f]))


class SerializeTests(unittest.TestCase):
    def test_roundtrip_digest(self):
        m = simple_module()
        j = to_json(m)
        m2 = from_json(j)
        self.assertEqual(digest(m), digest(m2))
        self.assertEqual(to_json(m2), j)
        self.assertIn("mul", disassemble(m))

    def test_strings_bytes_roundtrip(self):
        m = Module()
        m.intern_string(b"h\xe9llo\n\x00")
        m.functions.append(Function("main", 0, 0, [Block(0, [Instr(Op.INTR, sub=int(Intrinsic.WRITE_STR), imm=0), Instr(Op.RET)])]))
        self.assertEqual(from_json(to_json(m)).strings, m.strings)


class SemanticsTests(unittest.TestCase):
    def test_wrap(self):
        self.assertEqual(wrap64(1 << 63), -(1 << 63))
        self.assertEqual(wrap64((1 << 64) + 5), 5)
        self.assertEqual(binop(BinOp.ADD, (1 << 63) - 1, 1), -(1 << 63))
        self.assertEqual(binop(BinOp.MUL, 1 << 62, 4), 0)

    def test_division_semantics(self):
        self.assertEqual(binop(BinOp.DIV, -7, 2), -3)
        self.assertEqual(binop(BinOp.REM, -7, 2), -1)
        self.assertEqual(binop(BinOp.REM, 7, -2), 1)
        self.assertEqual(binop(BinOp.DIV, -(1 << 63), -1), -(1 << 63))
        self.assertEqual(binop(BinOp.REM, -(1 << 63), -1), 0)

    def test_shifts_and_logic(self):
        self.assertEqual(binop(BinOp.SHL, 1, 65), 2)
        self.assertEqual(binop(BinOp.SHR, -8, 1), -4)
        self.assertEqual(unop(UnOp.NOT, 0), 1)
        self.assertEqual(unop(UnOp.NOT, 9), 0)
        self.assertEqual(unop(UnOp.BNOT, 0), -1)
        self.assertEqual(unop(UnOp.NEG, -(1 << 63)), -(1 << 63))

    def test_trap(self):
        mb = ModuleBuilder()
        f = mb.function("main", 0)
        f.bin(BinOp.DIV, f.const(1), f.const(0))
        f.ret()
        r = Interpreter(mb.build()).run()
        self.assertEqual((r.exit_code, r.stderr), (TRAP_EXIT_CODE, TRAP_MESSAGE))

    def test_read_int_rules(self):
        mb = ModuleBuilder()
        f = mb.function("main", 0)
        for _ in range(5):
            v = f.intr(Intrinsic.READ_INT)
            f.intr(Intrinsic.WRITE_INT, [v])
            f.intr(Intrinsic.WRITE_CHAR, [f.const(32)])
        f.ret()
        r = Interpreter(mb.build(), stdin=b"  12\n-7\t+3 99999999999999999999 x5").run()
        # '+3' -> no digits -> 0 (consumes '+'), then '3' is read as the next number;
        # the huge literal wraps modulo 2**64; 'x5' -> 0
        self.assertEqual(r.stdout, b"12 -7 0 3 7766279631452241919 ")

    def test_exit_code_low_byte(self):
        mb = ModuleBuilder()
        f = mb.function("main", 0)
        f.ret(f.const(300))
        self.assertEqual(Interpreter(mb.build()).run().exit_code, 44)


if __name__ == "__main__":
    unittest.main()
