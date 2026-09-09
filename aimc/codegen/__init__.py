"""Code generation: IR -> raw, position-independent machine code.

    from aimc.codegen import compile_module, Target
    image = compile_module(module, Target.parse("aarch64-macos"))
    image.code            # bytes: [_start][aimc_entry][functions][runtime][rodata]
    image.start_offset    # process entry point (never returns; exits via syscall)
    image.entry_offset    # C-ABI `int64_t aimc_entry(void)` for in-process loaders
"""
from .image import Image, Assembler, AsmError
from .target import Target, host_target, SUPPORTED_TARGETS


def compile_module(module, target: Target) -> Image:
    from aimc.ir import verify
    verify(module)
    if target.arch == "x86_64":
        from .x86_64.backend import X86_64Backend as Backend
    elif target.arch == "aarch64":
        from .aarch64.backend import AArch64Backend as Backend
    else:  # pragma: no cover
        raise ValueError(f"unsupported architecture {target.arch}")
    return Backend(target).compile(module)


__all__ = ["compile_module", "Image", "Assembler", "AsmError", "Target", "host_target", "SUPPORTED_TARGETS"]
