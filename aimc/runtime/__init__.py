"""Execution runtime: container writers (ELF, Mach-O), an in-process JIT loader and a sandbox.

    from aimc.runtime import Sandbox, write_executable
    result = Sandbox().run(image, stdin=b"42\n")
"""
from .exe import write_executable, executable_suffix, code_vaddr
from .sandbox import Sandbox, RunResult, UnsupportedTarget

__all__ = ["write_executable", "executable_suffix", "code_vaddr", "Sandbox", "RunResult", "UnsupportedTarget"]
