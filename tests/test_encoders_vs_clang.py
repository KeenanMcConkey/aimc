"""Cross-check the encoders against the system assembler (clang) when one is available.

Each case pairs assembler text with the encoder call that should produce the
identical bytes.  Skipped entirely when clang/otool/objdump are missing.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest

from aimc.codegen.aarch64.encoder import A64Assembler, Cond, X
from aimc.codegen.x86_64.encoder import CC, R, X86Assembler

CLANG = shutil.which("clang")
OTOOL = shutil.which("otool")
OBJDUMP = shutil.which("objdump") or shutil.which("llvm-objdump")

A64_CASES = [
    ("movz x0, #42", lambda a: a.mov_ri(X.X0, 42)),
    ("movn x0, #0", lambda a: a.mov_ri(X.X0, -1)),
    ("movn x3, #1", lambda a: a.mov_ri(X.X3, -2)),
    ("movz x2, #0x6789\nmovk x2, #0x2345, lsl #16\nmovk x2, #1, lsl #32", lambda a: a.mov_ri(X.X2, 0x123456789)),
    ("movn x4, #0x1233\nmovk x4, #0xfffe, lsl #48", lambda a: a.mov_ri(X.X4, -0x0001000000001234)),
    ("mov x1, x2", lambda a: a.mov_rr(X.X1, X.X2)),
    ("add x29, sp, #0", lambda a: a.mov_sp(X.FP, X.SP)),
    ("add x0, x0, x1", lambda a: a.add_rr(X.X0, X.X0, X.X1)),
    ("sub x0, x0, x1", lambda a: a.sub_rr(X.X0, X.X0, X.X1)),
    ("and x0, x0, x1\norr x0, x0, x1\neor x0, x0, x1", lambda a: (a.and_rr(X.X0, X.X0, X.X1), a.orr_rr(X.X0, X.X0, X.X1), a.eor_rr(X.X0, X.X0, X.X1))),
    ("mvn x0, x0\nneg x0, x0", lambda a: (a.mvn(X.X0, X.X0), a.neg(X.X0, X.X0))),
    ("cmp x0, x1\ncmp x10, #0", lambda a: (a.cmp_rr(X.X0, X.X1), a.cmp_ri(X.X10, 0))),
    ("mul x0, x0, x1\nsdiv x0, x0, x1\nudiv x12, x0, x11", lambda a: (a.mul(X.X0, X.X0, X.X1), a.sdiv(X.X0, X.X0, X.X1), a.udiv(X.X12, X.X0, X.X11))),
    ("madd x19, x19, x9, x0\nmsub x13, x12, x11, x0", lambda a: (a.madd(X.X19, X.X19, X.X9, X.X0), a.msub(X.X13, X.X12, X.X11, X.X0))),
    ("lsl x0, x0, x1\nasr x0, x0, x1", lambda a: (a.lslv(X.X0, X.X0, X.X1), a.asrv(X.X0, X.X0, X.X1))),
    ("add x13, x13, #48\nsub sp, sp, #16\nsub sp, sp, x9", lambda a: (a.add_ri(X.X13, X.X13, 48), a.sub_ri(X.SP, X.SP, 16), a.sub_sp_reg(X.X9))),
    ("cset x0, eq\ncset x0, ne\ncset x0, lt\ncset x0, le\ncset x0, gt\ncset x0, ge",
     lambda a: [a.cset(X.X0, c) for c in (Cond.EQ, Cond.NE, Cond.LT, Cond.LE, Cond.GT, Cond.GE)]),
    ("ldr x0, [sp, #8]\nstr x1, [sp, #32752]\nldr x0, [sp, x15]\nstr x0, [sp, x15]",
     lambda a: (a.ldr(X.X0, X.SP, 8), a.str_(X.X1, X.SP, 32752), a.ldr_reg(X.X0, X.SP, X.X15), a.str_reg(X.X0, X.SP, X.X15))),
    ("ldrb w0, [sp]\nstrb w13, [x10]\nstrb w0, [sp, #16]", lambda a: (a.ldrb(X.X0, X.SP, 0), a.strb(X.X13, X.X10, 0), a.strb(X.X0, X.SP, 16))),
    ("stp x29, x30, [sp, #-48]!\nldp x29, x30, [sp], #48\nstp x19, x20, [sp, #16]\nldp x19, x20, [sp, #16]",
     lambda a: (a.stp_pre(X.FP, X.LR, X.SP, -48), a.ldp_post(X.FP, X.LR, X.SP, 48), a.stp(X.X19, X.X20, X.SP, 16), a.ldp(X.X19, X.X20, X.SP, 16))),
    ("b .+8\nbl .-4\ncbz x0, .-8\ncbnz x1, .+16\nb.le .-16\nadr x1, .-24\nret\nsvc #0x80\nsvc #0\nbrk #0",
     lambda a: (a.bind("s"), a.b(".p8"), a.bl(".m4"), a.cbz(X.X0, ".m8"), a.cbnz(X.X1, ".p16"), a.b_cond(Cond.LE, ".m16"), a.adr(X.X1, ".m24"),
                a.ret(), a.svc(0x80), a.svc(0), a.brk())),
]

X86_CASES = [
    ("mov eax, 42", lambda a: a.mov_ri32(R.RAX, 42)),
    ("mov rax, -1", lambda a: a.mov_ri(R.RAX, -1)),
    ("movabs r9, 0x1122334455667788", lambda a: a.mov_ri(R.R9, 0x1122334455667788)),
    ("mov rdi, rax\nmov r8, rdi\nmov rsi, r9", lambda a: (a.mov_rr(R.RDI, R.RAX), a.mov_rr(R.R8, R.RDI), a.mov_rr(R.RSI, R.R9))),
    ("mov rax, [rbp-8]\nmov [rbp-256], rcx\nmov rax, [rsp]\nmov rax, [r12+8]\nmov rax, [r13]\nmov [rbp-8000], r9",
     lambda a: (a.mov_rm(R.RAX, R.RBP, -8), a.mov_mr(R.RBP, -256, R.RCX), a.mov_rm(R.RAX, R.RSP, 0), a.mov_rm(R.RAX, R.R12, 8),
                a.mov_rm(R.RAX, R.R13, 0), a.mov_mr(R.RBP, -8000, R.R9))),
    ("movzx rax, byte ptr [rsp]\nmovzx rax, al\nmovzx rcx, cl", lambda a: (a.movzx_rm8(R.RAX, R.RSP, 0), a.movzx_rr8(R.RAX, R.RAX), a.movzx_rr8(R.RCX, R.RCX))),
    ("mov byte ptr [rsi], r9b\nmov byte ptr [rsi], 45", lambda a: (a.mov_m8r(R.RSI, 0, R.R9), a.mov_m8i(R.RSI, 0, 45))),
    ("lea rdi, [rip-17]", lambda a: (a.bind("t"), a.emit(b"\x90" * 10), a.lea_rip(R.RDI, "t"))),
    ("add rax, rcx\nsub rax, rcx\nand rax, rcx\nor rax, rcx\nxor rax, rcx\nxor eax, eax\ncmp rax, rcx\ntest r13, r13",
     lambda a: (a.add_rr(R.RAX, R.RCX), a.sub_rr(R.RAX, R.RCX), a.and_rr(R.RAX, R.RCX), a.or_rr(R.RAX, R.RCX), a.xor_rr(R.RAX, R.RCX),
                a.xor_rr32(R.RAX, R.RAX), a.cmp_rr(R.RAX, R.RCX), a.test_rr(R.R13, R.R13))),
    ("imul rax, rcx\nimul r12, r12, 10\nimul r12, r12, 1000", lambda a: (a.imul_rr(R.RAX, R.RCX), a.imul_rri(R.R12, R.R12, 10), a.imul_rri(R.R12, R.R12, 1000))),
    ("add rsp, 16\nsub rsp, 4096\nand rsp, -16\ncmp rcx, -1\ncmp rcx, 1000", lambda a: (a.add_ri(R.RSP, 16), a.sub_ri(R.RSP, 4096), a.and_ri(R.RSP, -16), a.cmp_ri(R.RCX, -1), a.cmp_ri(R.RCX, 1000))),
    ("not rax\nneg rax\ndiv rcx\nidiv rcx\ncqo\ninc r14\ndec rsi\nshl rax, cl\nsar rax, cl",
     lambda a: (a.not_r(R.RAX), a.neg_r(R.RAX), a.div_r(R.RCX), a.idiv_r(R.RCX), a.cqo(), a.inc_r(R.R14), a.dec_r(R.RSI), a.shl_cl(R.RAX), a.sar_cl(R.RAX))),
    # x86 relative branches are covered by X86EncoderTests.test_relative_fixups (clang's Intel
    # syntax has no portable spelling for a forced rel32 jump to a numeric offset).
    ("ret\npush rbp\npush r12\npop r12\npop rbp\nleave\nsyscall\nud2",
     lambda a: (a.ret(), a.push(R.RBP), a.push(R.R12), a.pop(R.R12), a.pop(R.RBP), a.leave(), a.syscall(), a.ud2())),
]


def _assemble(arch: str, text: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "t.s")
        obj = os.path.join(tmp, "t.o")
        prologue = ".intel_syntax noprefix\n" if arch == "x86_64" else ""
        with open(src, "w") as fh:
            fh.write(prologue + text + "\n")
        subprocess.run([CLANG, "-c", "-arch" if OTOOL else "--target=" + ("x86_64-linux-gnu" if arch == "x86_64" else "aarch64-linux-gnu"),
                        *([arch if arch == "x86_64" else "arm64"] if OTOOL else []), "-o", obj, src], check=True, capture_output=True)
        if OTOOL:
            out = subprocess.run([OTOOL, "-t", obj], capture_output=True, text=True, check=True).stdout
            hexes = []
            for line in out.splitlines()[2:]:
                parts = line.split()
                hexes.extend(parts[1:])
            if arch == "aarch64":
                return b"".join(bytes.fromhex(h)[::-1] for h in hexes)   # otool prints words big-endian
            return bytes.fromhex("".join(hexes))
        out = subprocess.run([OBJDUMP, "-d", obj], capture_output=True, text=True, check=True).stdout
        data = bytearray()
        for line in out.splitlines():
            m = re.match(r"\s*[0-9a-f]+:\s+((?:[0-9a-f]{2} )+|[0-9a-f]{8})", line)
            if m:
                chunk = m.group(1).strip().replace(" ", "")
                data += bytes.fromhex(chunk)[::-1] if arch == "aarch64" and len(chunk) == 8 else bytes.fromhex(chunk)
        return bytes(data)


def _bind_relative_labels(a, base_name: str = "s"):
    """Resolve the synthetic '.pN' / '.mN' / '.sN' labels used by branch cases relative to each instruction."""
    resolved = []
    for pos, kind, label in a.fixups:
        m = re.fullmatch(r"\.(p|m|s)(\d+)", label)
        if not m:
            resolved.append((pos, kind, label))
            continue
        instr_start = pos if kind != "rel32" else pos - (1 if kind == "rel32" and a.buf[pos - 1] in (0xE9, 0xE8) else 2)
        delta = int(m.group(2)) * (1 if m.group(1) == "p" else -1)
        if m.group(1) == "s":
            instr_start = pos - 1
            delta = -int(m.group(2))
        name = f"{label}@{pos}"
        a.labels[name] = instr_start + delta
        resolved.append((pos, kind, name))
    a.fixups = resolved


@unittest.skipUnless(CLANG and (OTOOL or OBJDUMP), "clang + otool/objdump not available")
class ClangCrossCheck(unittest.TestCase):
    def _run(self, arch, cases, asm_cls):
        for text, fn in cases:
            with self.subTest(asm=text.replace("\n", "; ")):
                a = asm_cls()
                fn(a)
                _bind_relative_labels(a)
                ours = a.finish()
                if text.startswith("lea rdi, [rip-17]"):
                    ours = ours[10:]  # strip the nop padding used to give the label a target
                theirs = _assemble(arch, text)
                self.assertEqual(ours.hex(), theirs.hex())

    def test_aarch64(self):
        self._run("aarch64", A64_CASES, A64Assembler)

    def test_x86_64(self):
        self._run("x86_64", X86_CASES, X86Assembler)


if __name__ == "__main__":
    unittest.main()
