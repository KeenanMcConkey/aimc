"""In-process JIT loader.

Maps a raw code image into executable memory and calls its C-ABI entry
point.  It is normally run as a *child process* by the sandbox (see
sandbox.py) so that a crashing or exiting program cannot take the compiler
down with it.  Usage as a script:

    python3 -m aimc.runtime.loader <image-file> [entry-offset]

Standard output / input of the loaded code are the process's own fds 0/1;
the exit status is the entry function's return value (or whatever the
program passed to the exit syscall).

This file must stay compatible with Python 3.8+ and use only the standard
library: on macOS it is frequently executed by /usr/bin/python3.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import platform
import sys

PROT_READ, PROT_WRITE, PROT_EXEC = 1, 2, 4
MAP_PRIVATE, MAP_ANON = 0x2, (0x1000 if sys.platform == "darwin" else 0x20)


def _libc():
    name = ctypes.util.find_library("c") or ("libc.so.6" if sys.platform != "darwin" else "libSystem.dylib")
    libc = ctypes.CDLL(name, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64]
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mprotect.restype = ctypes.c_int
    return libc


def load(code: bytes) -> int:
    """Map `code` R+X and return its base address."""
    libc = _libc()
    size = (len(code) + 0xFFFF) & ~0xFFFF
    addr = libc.mmap(None, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)
    if addr in (None, 0, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), "mmap failed")
    ctypes.memmove(addr, code, len(code))
    if libc.mprotect(addr, size, PROT_READ | PROT_EXEC) != 0:
        raise OSError(ctypes.get_errno(), "mprotect failed")
    _flush_icache(libc, addr, size)
    return addr


def _flush_icache(libc, addr: int, size: int) -> None:
    if platform.machine().lower() not in ("arm64", "aarch64"):
        return
    if sys.platform == "darwin":
        libc.sys_icache_invalidate.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.sys_icache_invalidate(addr, size)
        return
    for lib in ("gcc_s", "gcc"):  # pragma: no cover - Linux/aarch64 only
        path = ctypes.util.find_library(lib)
        if path:
            try:
                l = ctypes.CDLL(path)
                l.__clear_cache.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                l.__clear_cache(addr, addr + size)
                return
            except AttributeError:
                continue


def run(code: bytes, entry_offset: int) -> int:
    base = load(code)
    fn = ctypes.CFUNCTYPE(ctypes.c_int64)(base + entry_offset)
    sys.stdout.flush()
    sys.stderr.flush()
    return fn()


def main(argv) -> int:
    path = argv[1]
    entry = int(argv[2], 0) if len(argv) > 2 else 0
    with open(path, "rb") as fh:
        code = fh.read()
    return run(code, entry) & 0xFF


if __name__ == "__main__":
    sys.exit(main(sys.argv))
