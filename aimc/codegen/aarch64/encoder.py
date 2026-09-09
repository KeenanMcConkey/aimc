"""AArch64 (ARM64) instruction encoder: fixed 32-bit words, little-endian."""
from __future__ import annotations

import struct
from enum import IntEnum

from ..image import Assembler, AsmError, check_range


class X(IntEnum):
    X0 = 0
    X1 = 1
    X2 = 2
    X3 = 3
    X4 = 4
    X5 = 5
    X6 = 6
    X7 = 7
    X8 = 8
    X9 = 9
    X10 = 10
    X11 = 11
    X12 = 12
    X13 = 13
    X14 = 14
    X15 = 15
    X16 = 16
    X17 = 17
    X19 = 19
    X20 = 20
    X21 = 21
    FP = 29
    LR = 30
    SP = 31   # also XZR, depending on the instruction
    XZR = 31


class Cond(IntEnum):
    EQ = 0
    NE = 1
    CS = 2
    CC = 3
    MI = 4
    PL = 5
    VS = 6
    VC = 7
    HI = 8
    LS = 9
    GE = 10
    LT = 11
    GT = 12
    LE = 13
    AL = 14


class A64Assembler(Assembler):
    # -- fixups -----------------------------------------------------------
    def patch(self, kind: str, pos: int, target: int) -> None:
        rel = target - pos
        word = struct.unpack_from("<I", self.buf, pos)[0]
        if kind == "b26":
            check_range(rel // 4, 26, "b26")
            word |= (rel // 4) & 0x3FFFFFF
        elif kind == "b19":
            check_range(rel // 4, 19, "b19")
            word |= ((rel // 4) & 0x7FFFF) << 5
        elif kind == "adr21":
            check_range(rel, 21, "adr21")
            word |= ((rel & 3) << 29) | (((rel >> 2) & 0x7FFFF) << 5)
        else:  # pragma: no cover
            raise AsmError(f"unknown fixup kind {kind}")
        struct.pack_into("<I", self.buf, pos, word)

    def word(self, w: int) -> None:
        self.buf += struct.pack("<I", w & 0xFFFFFFFF)

    # -- moves ------------------------------------------------------------
    def mov_ri(self, rd: int, imm: int) -> None:
        """Materialise any 64-bit constant with MOVZ/MOVN + MOVK."""
        v = imm & 0xFFFFFFFFFFFFFFFF
        chunks = [(v >> (16 * i)) & 0xFFFF for i in range(4)]
        if chunks.count(0xFFFF) >= chunks.count(0) and chunks.count(0xFFFF) > 0:
            # MOVN-based: start from all-ones
            first = next(i for i, c in enumerate(chunks) if c != 0xFFFF) if any(c != 0xFFFF for c in chunks) else 0
            self.word(0x92800000 | (first << 21) | ((~chunks[first] & 0xFFFF) << 5) | rd)
            for i, c in enumerate(chunks):
                if i != first and c != 0xFFFF:
                    self.word(0xF2800000 | (i << 21) | (c << 5) | rd)
        else:
            first = next((i for i, c in enumerate(chunks) if c != 0), 0)
            self.word(0xD2800000 | (first << 21) | (chunks[first] << 5) | rd)
            for i, c in enumerate(chunks):
                if i != first and c != 0:
                    self.word(0xF2800000 | (i << 21) | (c << 5) | rd)

    def mov_rr(self, rd: int, rm: int) -> None:
        """ORR rd, xzr, rm  (use mov_sp for moves involving SP)."""
        self.word(0xAA0003E0 | (rm << 16) | rd)

    def mov_sp(self, rd: int, rn: int) -> None:
        """ADD rd, rn, #0 -- the form that accepts SP as either operand."""
        self.word(0x91000000 | (rn << 5) | rd)

    # -- arithmetic (shifted register) --------------------------------------
    def _dp3(self, base: int, rd: int, rn: int, rm: int) -> None:
        self.word(base | (rm << 16) | (rn << 5) | rd)

    def add_rr(self, rd: int, rn: int, rm: int) -> None:
        self._dp3(0x8B000000, rd, rn, rm)

    def sub_rr(self, rd: int, rn: int, rm: int) -> None:
        self._dp3(0xCB000000, rd, rn, rm)

    def and_rr(self, rd: int, rn: int, rm: int) -> None:
        self._dp3(0x8A000000, rd, rn, rm)

    def orr_rr(self, rd: int, rn: int, rm: int) -> None:
        self._dp3(0xAA000000, rd, rn, rm)

    def eor_rr(self, rd: int, rn: int, rm: int) -> None:
        self._dp3(0xCA000000, rd, rn, rm)

    def mvn(self, rd: int, rm: int) -> None:
        self._dp3(0xAA200000, rd, X.XZR, rm)

    def neg(self, rd: int, rm: int) -> None:
        self._dp3(0xCB000000, rd, X.XZR, rm)

    def cmp_rr(self, rn: int, rm: int) -> None:
        self._dp3(0xEB000000, X.XZR, rn, rm)

    def mul(self, rd: int, rn: int, rm: int) -> None:
        self.word(0x9B007C00 | (rm << 16) | (rn << 5) | rd)

    def madd(self, rd: int, rn: int, rm: int, ra: int) -> None:
        self.word(0x9B000000 | (rm << 16) | (ra << 10) | (rn << 5) | rd)

    def msub(self, rd: int, rn: int, rm: int, ra: int) -> None:
        self.word(0x9B008000 | (rm << 16) | (ra << 10) | (rn << 5) | rd)

    def sdiv(self, rd: int, rn: int, rm: int) -> None:
        self.word(0x9AC00C00 | (rm << 16) | (rn << 5) | rd)

    def udiv(self, rd: int, rn: int, rm: int) -> None:
        self.word(0x9AC00800 | (rm << 16) | (rn << 5) | rd)

    def lslv(self, rd: int, rn: int, rm: int) -> None:
        self.word(0x9AC02000 | (rm << 16) | (rn << 5) | rd)

    def asrv(self, rd: int, rn: int, rm: int) -> None:
        self.word(0x9AC02800 | (rm << 16) | (rn << 5) | rd)

    # -- arithmetic (immediate) ---------------------------------------------
    def _imm12(self, base: int, rd: int, rn: int, imm: int) -> None:
        check_range(imm, 12, "imm12", signed=False)
        self.word(base | (imm << 10) | (rn << 5) | rd)

    def add_ri(self, rd: int, rn: int, imm: int) -> None:
        self._imm12(0x91000000, rd, rn, imm)

    def sub_ri(self, rd: int, rn: int, imm: int) -> None:
        self._imm12(0xD1000000, rd, rn, imm)

    def cmp_ri(self, rn: int, imm: int) -> None:
        self._imm12(0xF1000000, X.XZR, rn, imm)

    def sub_sp_reg(self, rm: int) -> None:
        """SUB sp, sp, rm (extended-register form, required when SP is involved)."""
        self.word(0xCB2063FF | (rm << 16))

    def cset(self, rd: int, cond: int) -> None:
        self.word(0x9A9F07E0 | ((cond ^ 1) << 12) | rd)

    # -- memory ---------------------------------------------------------------
    def ldr(self, rt: int, rn: int, offset: int) -> None:
        if offset % 8 or not 0 <= offset <= 32760:
            raise AsmError(f"ldr offset {offset} not encodable")
        self.word(0xF9400000 | ((offset // 8) << 10) | (rn << 5) | rt)

    def str_(self, rt: int, rn: int, offset: int) -> None:
        if offset % 8 or not 0 <= offset <= 32760:
            raise AsmError(f"str offset {offset} not encodable")
        self.word(0xF9000000 | ((offset // 8) << 10) | (rn << 5) | rt)

    def ldr_reg(self, rt: int, rn: int, rm: int) -> None:
        """LDR rt, [rn, rm]  (register offset, no shift)."""
        self.word(0xF8606800 | (rm << 16) | (rn << 5) | rt)

    def str_reg(self, rt: int, rn: int, rm: int) -> None:
        """STR rt, [rn, rm]"""
        self.word(0xF8206800 | (rm << 16) | (rn << 5) | rt)

    def ldrb(self, rt: int, rn: int, offset: int = 0) -> None:
        check_range(offset, 12, "ldrb offset", signed=False)
        self.word(0x39400000 | (offset << 10) | (rn << 5) | rt)

    def strb(self, rt: int, rn: int, offset: int = 0) -> None:
        check_range(offset, 12, "strb offset", signed=False)
        self.word(0x39000000 | (offset << 10) | (rn << 5) | rt)

    def stp_pre(self, rt: int, rt2: int, rn: int, offset: int) -> None:
        """STP rt, rt2, [rn, #offset]!"""
        check_range(offset // 8, 7, "stp imm7")
        self.word(0xA9800000 | ((offset // 8 & 0x7F) << 15) | (rt2 << 10) | (rn << 5) | rt)

    def ldp_post(self, rt: int, rt2: int, rn: int, offset: int) -> None:
        """LDP rt, rt2, [rn], #offset"""
        check_range(offset // 8, 7, "ldp imm7")
        self.word(0xA8C00000 | ((offset // 8 & 0x7F) << 15) | (rt2 << 10) | (rn << 5) | rt)

    def stp(self, rt: int, rt2: int, rn: int, offset: int) -> None:
        check_range(offset // 8, 7, "stp imm7")
        self.word(0xA9000000 | ((offset // 8 & 0x7F) << 15) | (rt2 << 10) | (rn << 5) | rt)

    def ldp(self, rt: int, rt2: int, rn: int, offset: int) -> None:
        check_range(offset // 8, 7, "ldp imm7")
        self.word(0xA9400000 | ((offset // 8 & 0x7F) << 15) | (rt2 << 10) | (rn << 5) | rt)

    # -- control flow -------------------------------------------------------
    def b(self, label: str) -> None:
        self.fixup("b26", label)
        self.word(0x14000000)

    def bl(self, label: str) -> None:
        self.fixup("b26", label)
        self.word(0x94000000)

    def b_cond(self, cond: int, label: str) -> None:
        self.fixup("b19", label)
        self.word(0x54000000 | cond)

    def cbz(self, rt: int, label: str) -> None:
        self.fixup("b19", label)
        self.word(0xB4000000 | rt)

    def cbnz(self, rt: int, label: str) -> None:
        self.fixup("b19", label)
        self.word(0xB5000000 | rt)

    def adr(self, rd: int, label: str) -> None:
        self.fixup("adr21", label)
        self.word(0x10000000 | rd)

    def ret(self) -> None:
        self.word(0xD65F03C0)

    def svc(self, imm: int) -> None:
        self.word(0xD4000001 | (imm << 5))

    def brk(self, imm: int = 0) -> None:
        self.word(0xD4200000 | (imm << 5))
