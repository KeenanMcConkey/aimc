"""Dispatch to the executable container writer for an image's target OS."""
from __future__ import annotations

from aimc.codegen.image import Image


def write_executable(image: Image) -> bytes:
    if image.target.os == "linux":
        from .elf import write_elf
        return write_elf(image)
    if image.target.os == "macos":
        from .macho import write_macho
        return write_macho(image)
    raise ValueError(f"no executable format for {image.target.os}")


def executable_suffix(image: Image) -> str:
    return {"linux": ".elf", "macos": ".macho"}[image.target.os]


def code_vaddr(image: Image) -> int:
    """Virtual address at which `image.code[0]` is loaded by the container written by write_executable."""
    if image.target.os == "linux":
        from .elf import BASE, CODE_OFFSET
        return BASE + CODE_OFFSET
    from .macho import BASE, CODE_OFFSET
    return BASE + CODE_OFFSET
