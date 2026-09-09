"""x86_64 instruction encoder (raw bytes, no assembler in the loop).

Only the subset needed by the backend is implemented, but every method
encodes the full REX / ModRM / SIB machinery so any register combination
(including rsp/rbp/r12/r13 quirks) is handled correctly.
"""
from __future__ import annotations

import struct
from enum import IntEnum

from ..image import Assembler, AsmError, check_range


class R(IntEnum):
    RAX = 0
    RCX = 1
    RDX = 2
    RBX = 3
    RSP = 4
    RBP = 5
    RSI = 6
    RDI = 7
    R8 = 8
    R9 = 9
    R10 = 10
    R11 = 11
    R12 = 12
    R13 = 13
    R14 = 14
    R15 = 15


class CC(IntEnum):
    """Condition codes as used by Jcc / SETcc (low nibble of the opcode)."""
    O = 0x0
    NO = 0x1
    B = 0x2
    AE = 0x3
    E = 0x4
    NE = 0x5
    BE = 0x6
    A = 0x7
    S = 0x8
    NS = 0x9
    L = 0xC
    GE = 0xD
    LE = 0xE
    G = 0xF


def _fits8(v: int) -> bool:
    return -128 <= v <= 127


def _fits32(v: int) -> bool:
    return -(1 << 31) <= v < (1 << 31)


class X86Assembler(Assembler):
    # -- fixups -----------------------------------------------------------
    def patch(self, kind: str, pos: int, target: int) -> None:
        if kind == "rel32":
            rel = target - (pos + 4)
            check_range(rel, 32, "rel32")
            self.buf[pos:pos + 4] = struct.pack("<i", rel)
        else:  # pragma: no cover
            raise AsmError(f"unknown fixup kind {kind}")

    # -- encoding helpers -------------------------------------------------
    def _rex(self, w: int, reg: int, index: int, base: int, force: bool = False) -> None:
        rex = 0x40 | (w << 3) | ((reg >> 3) << 2) | ((index >> 3) << 1) | (base >> 3)
        if rex != 0x40 or force:
            self.buf.append(rex)

    def _modrm_reg(self, reg: int, rm: int) -> None:
        self.buf.append(0xC0 | ((reg & 7) << 3) | (rm & 7))

    def _modrm_mem(self, reg: int, base: int, disp: int) -> None:
        """[base + disp] addressing with correct SIB / disp handling."""
        needs_sib = (base & 7) == R.RSP
        if disp == 0 and (base & 7) != R.RBP:
            mod = 0
        elif _fits8(disp):
            mod = 1
        else:
            mod = 2
        self.buf.append((mod << 6) | ((reg & 7) << 3) | (0x4 if needs_sib else (base & 7)))
        if needs_sib:
            self.buf.append(0x24)  # scale=0, index=none(rsp), base=rsp/r12
        if mod == 1:
            self.buf += struct.pack("<b", disp)
        elif mod == 2:
            check_range(disp, 32, "disp32")
            self.buf += struct.pack("<i", disp)

    def _op_rr(self, opcode: int, reg: int, rm: int, w: int = 1, prefix: bytes = b"") -> None:
        self.buf += prefix
        self._rex(w, reg, 0, rm)
        self.buf.append(opcode)
        self._modrm_reg(reg, rm)

    def _op_rr2(self, op1: int, op2: int, reg: int, rm: int) -> None:
        self._rex(1, reg, 0, rm)
        self.buf += bytes([op1, op2])
        self._modrm_reg(reg, rm)

    def _op_rm(self, opcode: int, reg: int, base: int, disp: int, w: int = 1, force_rex: bool = False) -> None:
        self._rex(w, reg, 0, base, force=force_rex)
        self.buf.append(opcode)
        self._modrm_mem(reg, base, disp)

    # -- moves ------------------------------------------------------------
    def mov_rr(self, dst: int, src: int) -> None:
        self._op_rr(0x89, src, dst)

    def mov_ri(self, dst: int, imm: int) -> None:
        if imm < 0:
            imm_s = imm
        else:
            imm_s = imm - (1 << 64) if imm >> 63 else imm
        if _fits32(imm_s):
            self._rex(1, 0, 0, dst)
            self.buf.append(0xC7)
            self._modrm_reg(0, dst)
            self.buf += struct.pack("<i", imm_s)
        else:
            self._rex(1, 0, 0, dst)
            self.buf.append(0xB8 + (dst & 7))
            self.buf += struct.pack("<q", imm_s)

    def mov_ri32(self, dst: int, imm: int) -> None:
        """mov r32, imm32 (zero-extends into the 64-bit register)."""
        check_range(imm, 32, "imm32", signed=False)
        self._rex(0, 0, 0, dst)
        self.buf.append(0xB8 + (dst & 7))
        self.buf += struct.pack("<I", imm)

    def mov_rm(self, dst: int, base: int, disp: int) -> None:
        self._op_rm(0x8B, dst, base, disp)

    def mov_mr(self, base: int, disp: int, src: int) -> None:
        self._op_rm(0x89, src, base, disp)

    def mov_m8r(self, base: int, disp: int, src: int) -> None:
        """mov byte [base+disp], src8 (REX forced so sil/dil/… are addressable)."""
        self._op_rm(0x88, src, base, disp, w=0, force_rex=True)

    def mov_m8i(self, base: int, disp: int, imm: int) -> None:
        self._rex(0, 0, 0, base)
        self.buf.append(0xC6)
        self._modrm_mem(0, base, disp)
        self.buf.append(imm & 0xFF)

    def movzx_rm8(self, dst: int, base: int, disp: int) -> None:
        self._rex(1, dst, 0, base)
        self.buf += b"\x0f\xb6"
        self._modrm_mem(dst, base, disp)

    def movzx_rr8(self, dst: int, src: int) -> None:
        self._rex(1, dst, 0, src, force=True)
        self.buf += b"\x0f\xb6"
        self._modrm_reg(dst, src)

    def lea_rm(self, dst: int, base: int, disp: int) -> None:
        self._op_rm(0x8D, dst, base, disp)

    def lea_rip(self, dst: int, label: str) -> None:
        self._rex(1, dst, 0, 0)
        self.buf.append(0x8D)
        self.buf.append(0x05 | ((dst & 7) << 3))  # mod=00 rm=101 -> RIP+disp32
        self.fixup("rel32", label)
        self.buf += b"\0\0\0\0"

    # -- arithmetic -------------------------------------------------------
    def add_rr(self, dst: int, src: int) -> None:
        self._op_rr(0x01, src, dst)

    def sub_rr(self, dst: int, src: int) -> None:
        self._op_rr(0x29, src, dst)

    def and_rr(self, dst: int, src: int) -> None:
        self._op_rr(0x21, src, dst)

    def or_rr(self, dst: int, src: int) -> None:
        self._op_rr(0x09, src, dst)

    def xor_rr(self, dst: int, src: int) -> None:
        self._op_rr(0x31, src, dst)

    def xor_rr32(self, dst: int, src: int) -> None:
        self._op_rr(0x31, src, dst, w=0)

    def cmp_rr(self, a: int, b: int) -> None:
        self._op_rr(0x39, b, a)

    def test_rr(self, a: int, b: int) -> None:
        self._op_rr(0x85, b, a)

    def imul_rr(self, dst: int, src: int) -> None:
        self._op_rr2(0x0F, 0xAF, dst, src)

    def imul_rri(self, dst: int, src: int, imm: int) -> None:
        self._rex(1, dst, 0, src)
        if _fits8(imm):
            self.buf.append(0x6B)
            self._modrm_reg(dst, src)
            self.buf += struct.pack("<b", imm)
        else:
            self.buf.append(0x69)
            self._modrm_reg(dst, src)
            self.buf += struct.pack("<i", imm)

    def _grp1(self, ext: int, dst: int, imm: int) -> None:
        self._rex(1, 0, 0, dst)
        if _fits8(imm):
            self.buf.append(0x83)
            self._modrm_reg(ext, dst)
            self.buf += struct.pack("<b", imm)
        else:
            check_range(imm, 32, "imm32")
            self.buf.append(0x81)
            self._modrm_reg(ext, dst)
            self.buf += struct.pack("<i", imm)

    def add_ri(self, dst: int, imm: int) -> None:
        self._grp1(0, dst, imm)

    def sub_ri(self, dst: int, imm: int) -> None:
        self._grp1(5, dst, imm)

    def and_ri(self, dst: int, imm: int) -> None:
        self._grp1(4, dst, imm)

    def cmp_ri(self, dst: int, imm: int) -> None:
        self._grp1(7, dst, imm)

    def _grp3(self, ext: int, r: int) -> None:
        self._rex(1, 0, 0, r)
        self.buf.append(0xF7)
        self._modrm_reg(ext, r)

    def not_r(self, r: int) -> None:
        self._grp3(2, r)

    def neg_r(self, r: int) -> None:
        self._grp3(3, r)

    def div_r(self, r: int) -> None:
        self._grp3(6, r)

    def idiv_r(self, r: int) -> None:
        self._grp3(7, r)

    def cqo(self) -> None:
        self.buf += b"\x48\x99"

    def inc_r(self, r: int) -> None:
        self._rex(1, 0, 0, r)
        self.buf.append(0xFF)
        self._modrm_reg(0, r)

    def dec_r(self, r: int) -> None:
        self._rex(1, 0, 0, r)
        self.buf.append(0xFF)
        self._modrm_reg(1, r)

    def shl_cl(self, r: int) -> None:
        self._rex(1, 0, 0, r)
        self.buf.append(0xD3)
        self._modrm_reg(4, r)

    def sar_cl(self, r: int) -> None:
        self._rex(1, 0, 0, r)
        self.buf.append(0xD3)
        self._modrm_reg(7, r)

    def setcc(self, cc: int, r8: int) -> None:
        self._rex(0, 0, 0, r8, force=True)
        self.buf += bytes([0x0F, 0x90 | cc])
        self._modrm_reg(0, r8)

    # -- control flow -----------------------------------------------------
    def jmp(self, label: str) -> None:
        self.buf.append(0xE9)
        self.fixup("rel32", label)
        self.buf += b"\0\0\0\0"

    def jcc(self, cc: int, label: str) -> None:
        self.buf += bytes([0x0F, 0x80 | cc])
        self.fixup("rel32", label)
        self.buf += b"\0\0\0\0"

    def call(self, label: str) -> None:
        self.buf.append(0xE8)
        self.fixup("rel32", label)
        self.buf += b"\0\0\0\0"

    def ret(self) -> None:
        self.buf.append(0xC3)

    def push(self, r: int) -> None:
        self._rex(0, 0, 0, r)
        self.buf.append(0x50 + (r & 7))

    def pop(self, r: int) -> None:
        self._rex(0, 0, 0, r)
        self.buf.append(0x58 + (r & 7))

    def leave(self) -> None:
        self.buf.append(0xC9)

    def syscall(self) -> None:
        self.buf += b"\x0f\x05"

    def ud2(self) -> None:
        self.buf += b"\x0f\x0b"
