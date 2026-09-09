"""Target descriptions: architecture + operating system + syscall ABI."""
from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Target:
    arch: str   # "x86_64" | "aarch64"
    os: str     # "linux" | "macos"

    @property
    def name(self) -> str:
        return f"{self.arch}-{self.os}"

    @property
    def syscalls(self) -> Dict[str, int]:
        return _SYSCALLS[(self.arch, self.os)]

    @property
    def page_size(self) -> int:
        return 0x4000 if (self.arch, self.os) == ("aarch64", "macos") else 0x1000

    @staticmethod
    def parse(name: str) -> "Target":
        arch, _, os_ = name.partition("-")
        arch = _ARCH_ALIASES.get(arch, arch)
        os_ = _OS_ALIASES.get(os_, os_)
        if (arch, os_) not in _SYSCALLS:
            raise ValueError(f"unsupported target {name!r}; choose from {', '.join(SUPPORTED_TARGETS)}")
        return Target(arch, os_)

    def __str__(self) -> str:
        return self.name


_ARCH_ALIASES = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64", "arm64e": "aarch64"}
_OS_ALIASES = {"darwin": "macos", "osx": "macos"}

# Raw system call numbers.  macOS x86_64 numbers carry the 0x2000000 "UNIX class" prefix.
_SYSCALLS = {
    ("x86_64", "linux"): {"read": 0, "write": 1, "exit": 60},
    ("aarch64", "linux"): {"read": 63, "write": 64, "exit": 93},
    ("x86_64", "macos"): {"read": 0x2000003, "write": 0x2000004, "exit": 0x2000001},
    ("aarch64", "macos"): {"read": 3, "write": 4, "exit": 1},
}

SUPPORTED_TARGETS = tuple(f"{a}-{o}" for a, o in _SYSCALLS)


def host_arch() -> str:
    m = platform.machine().lower()
    return _ARCH_ALIASES.get(m, m)


def host_os() -> str:
    return {"darwin": "macos", "linux": "linux"}.get(sys.platform, sys.platform)


def host_target() -> Target:
    """The target matching the *machine* (not the Python interpreter, which may run under Rosetta)."""
    arch = host_arch()
    if sys.platform == "darwin":
        try:
            import subprocess
            out = subprocess.run(["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True, timeout=5)
            if out.stdout.strip() == "1":
                arch = "aarch64"
        except Exception:  # pragma: no cover
            pass
    return Target(arch, host_os())
