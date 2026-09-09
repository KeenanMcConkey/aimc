"""Position-independent code images and the label/fixup assembler core shared by all backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from .target import Target


class AsmError(Exception):
    pass


@dataclass
class Image:
    target: Target
    code: bytes
    start_offset: int
    entry_offset: int
    symbols: Dict[str, int] = field(default_factory=dict)
    rodata_offset: int = 0

    def symbol_at(self, offset: int) -> str:
        best = ""
        best_off = -1
        for name, off in self.symbols.items():
            if best_off < off <= offset:
                best, best_off = name, off
        return f"{best}+{offset - best_off:#x}" if best else f"+{offset:#x}"


class Assembler:
    """Byte buffer with labels and PC-relative fixups.

    Subclasses implement `patch(kind, pos, target)` to encode a resolved
    relative offset into the instruction bytes at `pos`.
    """

    def __init__(self) -> None:
        self.buf = bytearray()
        self.labels: Dict[str, int] = {}
        self.fixups: List[Tuple[int, str, str]] = []  # (position, kind, label)
        self._uniq = 0

    # -- positions --------------------------------------------------------
    @property
    def pos(self) -> int:
        return len(self.buf)

    def bind(self, name: str) -> int:
        if name in self.labels:
            raise AsmError(f"label {name!r} bound twice")
        self.labels[name] = self.pos
        return self.pos

    def local(self, hint: str = "L") -> str:
        self._uniq += 1
        return f".{hint}{self._uniq}"

    def emit(self, data: bytes) -> None:
        self.buf += data

    def align(self, n: int, fill: bytes = b"\0") -> None:
        while self.pos % n:
            self.buf += fill[: n - self.pos % n] if len(fill) > 1 else fill

    def fixup(self, kind: str, label: str, pos: int = -1) -> None:
        self.fixups.append((self.pos if pos < 0 else pos, kind, label))

    # -- resolution -------------------------------------------------------
    def patch(self, kind: str, pos: int, target: int) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def finish(self) -> bytes:
        for pos, kind, label in self.fixups:
            if label not in self.labels:
                raise AsmError(f"undefined label {label!r}")
            self.patch(kind, pos, self.labels[label])
        return bytes(self.buf)


def check_range(value: int, bits: int, what: str, signed: bool = True) -> None:
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    if not lo <= value <= hi:
        raise AsmError(f"{what}: value {value} does not fit in {bits} bits")
