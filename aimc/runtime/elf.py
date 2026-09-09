"""Minimal static ELF64 executable writer (Linux x86_64 / aarch64).

One PT_LOAD segment maps the whole file R+X at BASE; the entry point is the
image's `_start` stub.  No sections, no dynamic linker, no libc.
"""
from __future__ import annotations

import struct

from aimc.codegen.image import Image

BASE = 0x400000
EM = {"x86_64": 0x3E, "aarch64": 0xB7}
CODE_OFFSET = 0x80  # right after the ELF and program headers (16-aligned)


def write_elf(image: Image) -> bytes:
    if image.target.os != "linux":
        raise ValueError("ELF images are only produced for linux targets")
    code = image.code
    filesize = CODE_OFFSET + len(code)
    entry = BASE + CODE_OFFSET + image.start_offset
    ehdr = struct.pack(
        "<4sBBBBB7xHHIQQQIHHHHHH",
        b"\x7fELF", 2, 1, 1, 0, 0,          # 64-bit, little-endian, version 1, SYSV ABI
        2, EM[image.target.arch], 1,        # ET_EXEC, machine, version
        entry, 64, 0,                       # e_entry, e_phoff, e_shoff
        0, 64, 56, 1, 64, 0, 0,             # flags, ehsize, phentsize, phnum, shentsize, shnum, shstrndx
    )
    phdr = struct.pack(
        "<IIQQQQQQ",
        1, 5,                               # PT_LOAD, PF_R | PF_X
        0, BASE, BASE,                      # offset, vaddr, paddr
        filesize, filesize, 0x1000,         # filesz, memsz, align
    )
    hdr = ehdr + phdr
    return hdr + b"\0" * (CODE_OFFSET - len(hdr)) + code
