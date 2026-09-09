"""Machine-oriented intermediate representation (IR) for aimc.

Public surface:
    nodes      - Module / Function / Block / Instr and opcode enums
    builder    - ergonomic construction of IR
    verify     - structural + dataflow verification
    serialize  - canonical JSON encoding, digests, disassembly
    interp     - reference interpreter (defines the semantics)
"""
from .nodes import (Module, Function, Block, Instr, Op, BinOp, UnOp, Intrinsic,
                    TRAP_EXIT_CODE, TRAP_MESSAGE, wrap64)
from .builder import ModuleBuilder, FunctionBuilder
from .verify import verify, IRError
from .serialize import to_json, from_json, digest, disassemble
from .interp import Interpreter, ExecResult

__all__ = ["Module", "Function", "Block", "Instr", "Op", "BinOp", "UnOp", "Intrinsic",
           "TRAP_EXIT_CODE", "TRAP_MESSAGE", "wrap64", "ModuleBuilder", "FunctionBuilder",
           "verify", "IRError", "to_json", "from_json", "digest", "disassemble",
           "Interpreter", "ExecResult"]
