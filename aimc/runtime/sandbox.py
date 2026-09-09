"""Sandboxed execution of compiled images.

Two execution strategies:

  * "exe"  - write a native executable (ELF / signed Mach-O) to a scratch
             directory and run it as a subprocess.  This is the real thing:
             the kernel loads the bytes the compiler produced.
  * "jit"  - run `aimc.runtime.loader` in a child Python process whose
             architecture matches the image (on Apple Silicon `/usr/bin/arch`
             selects the slice); the loader mmaps the raw code and calls it.

Both run in a fresh, empty working directory with a minimal environment,
a wall-clock timeout and CPU / address-space rlimits, with stdin fed from a
byte string and stdout/stderr captured.
"""
from __future__ import annotations

import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from aimc.codegen.image import Image
from aimc.codegen.target import Target, host_os, host_target

from .exe import write_executable, executable_suffix
from .loader import __file__ as LOADER_PATH


class UnsupportedTarget(Exception):
    pass


@dataclass
class RunResult:
    stdout: bytes
    stderr: bytes
    exit_code: int
    timed_out: bool
    strategy: str
    signal: Optional[int] = None

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.signal is None


class Sandbox:
    def __init__(self, timeout: float = 5.0, memory_limit: int = 256 << 20, cpu_seconds: int = 5):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_seconds = cpu_seconds

    # -- capability probing ------------------------------------------------
    def strategies(self, target: Target) -> List[str]:
        """Strategies able to execute `target` on this machine, best first."""
        out: List[str] = []
        if target.os != host_os():
            return out
        if _can_exec_arch(target.arch):
            out.append("exe")
        if _python_for_arch(target.arch):
            out.append("jit")
        return out

    def run(self, image: Image, stdin: bytes = b"", strategy: str = "auto") -> RunResult:
        avail = self.strategies(image.target)
        if strategy == "auto":
            if not avail:
                raise UnsupportedTarget(f"cannot execute {image.target} on this {host_target()} host")
            strategy = avail[0]
        elif strategy not in avail:
            raise UnsupportedTarget(f"strategy {strategy!r} unavailable for {image.target} (have: {avail or 'none'})")
        with tempfile.TemporaryDirectory(prefix="aimc-run-") as tmp:
            if strategy == "exe":
                exe = os.path.join(tmp, "program" + executable_suffix(image))
                with open(exe, "wb") as fh:
                    fh.write(write_executable(image))
                os.chmod(exe, 0o755)
                argv = _arch_prefix(image.target.arch) + [exe]
            else:
                blob = os.path.join(tmp, "image.bin")
                with open(blob, "wb") as fh:
                    fh.write(image.code)
                argv = _arch_prefix(image.target.arch) + [_python_for_arch(image.target.arch), LOADER_PATH,
                                                          blob, str(image.entry_offset)]
            return self._spawn(argv, stdin, tmp, strategy)

    # -- process control ---------------------------------------------------
    def _spawn(self, argv: List[str], stdin: bytes, cwd: str, strategy: str) -> RunResult:
        limits = (self.cpu_seconds, self.memory_limit)

        def preexec() -> None:  # pragma: no cover - runs in the child
            cpu, mem = limits
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
            except (ValueError, OSError):
                pass
            if strategy == "exe" and sys.platform != "darwin":
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
                except (ValueError, OSError):
                    pass

        env = {"PATH": "/usr/bin:/bin", "LANG": "C"}
        try:
            proc = subprocess.run(argv, input=stdin, capture_output=True, timeout=self.timeout,
                                  cwd=cwd, env=env, preexec_fn=preexec)
        except subprocess.TimeoutExpired as e:
            return RunResult(e.stdout or b"", e.stderr or b"", -1, True, strategy)
        sig = -proc.returncode if proc.returncode < 0 else None
        return RunResult(proc.stdout, proc.stderr, proc.returncode & 0xFF if sig is None else -1, False, strategy, sig)


# -- host capability helpers ---------------------------------------------------
def _host_machine() -> str:
    return host_target().arch


def _arch_prefix(arch: str) -> List[str]:
    """On macOS, `/usr/bin/arch` selects the slice to run (native arm64 or Rosetta x86_64)."""
    if sys.platform == "darwin" and os.path.exists("/usr/bin/arch"):
        return ["/usr/bin/arch", "-arm64" if arch == "aarch64" else "-x86_64"]
    return []


@lru_cache(maxsize=None)
def _can_exec_arch(arch: str) -> bool:
    if arch == _host_machine():
        return True
    if sys.platform == "darwin" and arch == "x86_64":
        try:
            r = subprocess.run(["/usr/bin/arch", "-x86_64", "/usr/bin/true"], capture_output=True, timeout=10)
            return r.returncode == 0
        except Exception:
            return False
    return False


@lru_cache(maxsize=None)
def _python_for_arch(arch: str) -> Optional[str]:
    """A Python interpreter whose process architecture is `arch` (for the JIT loader)."""
    candidates = [sys.executable, "/usr/bin/python3", shutil.which("python3"), shutil.which("python3.11"),
                  shutil.which("python3.12"), shutil.which("python3.13")]
    seen = set()
    for c in candidates:
        if not c or c in seen or not os.path.exists(c):
            continue
        seen.add(c)
        try:
            r = subprocess.run(_arch_prefix(arch) + [c, "-c", "import platform;print(platform.machine())"],
                               capture_output=True, text=True, timeout=15)
        except Exception:
            continue
        m = r.stdout.strip().lower()
        if r.returncode == 0 and {"arm64": "aarch64", "amd64": "x86_64"}.get(m, m) == arch:
            return c
    return None
