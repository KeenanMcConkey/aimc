"""Minimal Mach-O executable writer for macOS (x86_64 and arm64) with ad-hoc code signing.

Apple Silicon refuses to run static (non-dyld) arm64 executables and refuses
unsigned arm64 code entirely, so the image is emitted as the smallest
dyld-linked executable the kernel accepts:

    __PAGEZERO, __TEXT (code + rodata), __LINKEDIT (symbol table + signature)
    LC_SYMTAB, LC_DYSYMTAB, LC_LOAD_DYLINKER, LC_BUILD_VERSION, LC_MAIN, LC_CODE_SIGNATURE

No libraries are linked: the program talks to the kernel through raw
syscalls.  The ad-hoc signature (a SHA-256 CodeDirectory) is generated here in
pure Python, so no `codesign` tool is required.
"""
from __future__ import annotations

import hashlib
import struct

from aimc.codegen.image import Image

BASE = 0x100000000
CPU = {"x86_64": (0x01000007, 0x3), "aarch64": (0x0100000C, 0x0)}

MH_MAGIC_64 = 0xFEEDFACF
MH_EXECUTE = 2
MH_NOUNDEFS, MH_DYLDLINK, MH_TWOLEVEL, MH_PIE = 0x1, 0x4, 0x80, 0x200000
LC_SEGMENT_64, LC_SYMTAB, LC_DYSYMTAB, LC_LOAD_DYLINKER = 0x19, 0x2, 0xB, 0xE
LC_BUILD_VERSION, LC_MAIN, LC_CODE_SIGNATURE = 0x32, 0x80000028, 0x1D
CODE_OFFSET = 0x1000  # code starts here inside __TEXT; leaves room for the load commands


def _lc(cmd: int, body: bytes) -> bytes:
    body += b"\0" * ((-len(body)) % 8)
    return struct.pack("<II", cmd, 8 + len(body)) + body


def _segment(name: bytes, vmaddr: int, vmsize: int, fileoff: int, filesize: int,
             maxprot: int, initprot: int, sections: bytes = b"", nsects: int = 0) -> bytes:
    return struct.pack("<II16sQQQQIIII", LC_SEGMENT_64, 72 + len(sections), name, vmaddr, vmsize,
                       fileoff, filesize, maxprot, initprot, nsects, 0) + sections


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def write_macho(image: Image, identifier: str = "aimc") -> bytes:
    if image.target.os != "macos":
        raise ValueError("Mach-O images are only produced for macos targets")
    page = image.target.page_size
    cputype, cpusubtype = CPU[image.target.arch]
    code = image.code
    text_size = _align(CODE_OFFSET + len(code), page)
    linkedit_off = text_size
    symtab = b"\0" * 8                       # empty string table (offset 0 is the empty string)
    sig_off = linkedit_off + _align(len(symtab), 16)
    sig = _adhoc_signature(b"", sig_off, text_size, identifier)  # placeholder for size
    sig_size = _align(len(sig), 16)
    linkedit_size = (sig_off - linkedit_off) + sig_size

    text_sect = struct.pack("<16s16sQQIIIIIIII", b"__text", b"__TEXT", BASE + CODE_OFFSET, len(code),
                            CODE_OFFSET, 4, 0, 0, 0x80000400, 0, 0, 0)
    cmds = [
        _segment(b"__PAGEZERO", 0, BASE, 0, 0, 0, 0),
        _segment(b"__TEXT", BASE, text_size, 0, text_size, 5, 5, text_sect, 1),
        _segment(b"__LINKEDIT", BASE + text_size, _align(linkedit_size, page), linkedit_off, linkedit_size, 1, 1),
        _lc(LC_SYMTAB, struct.pack("<IIII", linkedit_off, 0, linkedit_off, len(symtab))),
        _lc(LC_DYSYMTAB, struct.pack("<18I", *([0] * 18))),
        _lc(LC_LOAD_DYLINKER, struct.pack("<I", 12) + b"/usr/lib/dyld\0"),
        _lc(LC_BUILD_VERSION, struct.pack("<IIII", 1, 0x000B0000, 0x000E0000, 0)),  # macOS, minos 11.0, sdk 14.0
        _lc(LC_MAIN, struct.pack("<QQ", CODE_OFFSET + image.start_offset, 0)),
        _lc(LC_CODE_SIGNATURE, struct.pack("<II", sig_off, sig_size)),
    ]
    body = b"".join(cmds)
    flags = MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL | MH_PIE
    header = struct.pack("<IIIIIIII", MH_MAGIC_64, cputype, cpusubtype, MH_EXECUTE, len(cmds), len(body), flags, 0)
    out = bytearray(header + body)
    if len(out) > CODE_OFFSET:
        raise ValueError("load commands overflow the header page")
    out += b"\0" * (CODE_OFFSET - len(out)) + code
    out += b"\0" * (text_size - len(out))
    out += symtab
    out += b"\0" * (sig_off - len(out))
    sig = _adhoc_signature(bytes(out), sig_off, text_size, identifier)
    out += sig + b"\0" * (sig_size - len(sig))
    return bytes(out)


# -- ad-hoc code signature ---------------------------------------------------
CSMAGIC_EMBEDDED_SIGNATURE = 0xFADE0CC0
CSMAGIC_CODEDIRECTORY = 0xFADE0C02
CS_ADHOC = 0x2
CS_EXECSEG_MAIN_BINARY = 0x1
CS_PAGE_SHIFT = 12


def _adhoc_signature(data: bytes, code_limit: int, text_size: int, identifier: str) -> bytes:
    """Build an embedded signature SuperBlob containing one SHA-256 CodeDirectory."""
    page = 1 << CS_PAGE_SHIFT
    hashes = b"".join(hashlib.sha256(data[i:i + page]).digest() for i in range(0, code_limit, page))
    n_slots = (code_limit + page - 1) // page
    ident = identifier.encode() + b"\0"
    header_len = 88
    ident_off = header_len
    hash_off = ident_off + len(ident)
    cd_len = hash_off + len(hashes)
    cd = struct.pack(
        ">IIIIIIIIIBBBBIIIIQQQQ",
        CSMAGIC_CODEDIRECTORY, cd_len, 0x20400, CS_ADHOC,
        hash_off, ident_off, 0, n_slots, code_limit,
        32, 2, 0, CS_PAGE_SHIFT,          # hashSize, hashType=SHA256, platform, pageSize(log2)
        0,                                # spare2
        0, 0, 0,                          # scatterOffset, teamOffset, spare3
        0,                                # codeLimit64
        0, text_size, CS_EXECSEG_MAIN_BINARY,
    )
    assert len(cd) == header_len
    cd += ident + hashes
    blob_len = 12 + 8 + len(cd)
    return struct.pack(">III", CSMAGIC_EMBEDDED_SIGNATURE, blob_len, 1) + struct.pack(">II", 0, 20) + cd
