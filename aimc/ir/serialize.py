"""Canonical JSON encoding, content digests and a debugging disassembly."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from .nodes import BinOp, Block, Function, Instr, Intrinsic, Module, Op, UnOp

FORMAT_VERSION = 1


def to_dict(module: Module) -> Dict[str, Any]:
    return {
        "aimc_ir": FORMAT_VERSION,
        "entry": module.entry,
        "strings": [s.decode("latin-1") for s in module.strings],
        "functions": [
            {
                "name": f.name,
                "params": f.nparams,
                "slots": f.nslots,
                "blocks": [[ins.to_list() for ins in b.instrs] for b in f.blocks],
            }
            for f in module.functions
        ],
    }


def from_dict(d: Dict[str, Any]) -> Module:
    if d.get("aimc_ir") != FORMAT_VERSION:
        raise ValueError(f"unsupported IR format version {d.get('aimc_ir')!r}")
    m = Module(entry=int(d.get("entry", 0)))
    m.strings = [s.encode("latin-1") for s in d.get("strings", [])]
    for fd in d["functions"]:
        blocks = [Block(i, [Instr.from_list(raw) for raw in instrs]) for i, instrs in enumerate(fd["blocks"])]
        m.functions.append(Function(fd["name"], int(fd["params"]), int(fd["slots"]), blocks))
    return m


def to_json(module: Module, pretty: bool = False) -> str:
    d = to_dict(module)
    if pretty:
        return json.dumps(d, indent=1, sort_keys=True)
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def from_json(text: str) -> Module:
    return from_dict(json.loads(text))


def digest(module: Module) -> str:
    """SHA-256 of the canonical encoding; two modules with equal digests are semantically identical."""
    return hashlib.sha256(to_json(module).encode("ascii")).hexdigest()


def disassemble(module: Module) -> str:
    lines = [f"; aimc IR v{FORMAT_VERSION}  digest={digest(module)[:16]}"]
    for i, s in enumerate(module.strings):
        lines.append(f"str{i} = {s!r}")
    for fi, f in enumerate(module.functions):
        tag = " (entry)" if fi == module.entry else ""
        lines.append(f"fn{fi} {f.name}(params={f.nparams}, slots={f.nslots}){tag}:")
        for b in f.blocks:
            lines.append(f"  b{b.id}:")
            for ins in b.instrs:
                lines.append("    " + _fmt(module, ins))
    return "\n".join(lines) + "\n"


def _fmt(module: Module, ins: Instr) -> str:
    s = lambda x: f"s{x}"
    if ins.op == Op.CONST:
        return f"{s(ins.dst)} = const {ins.imm}"
    if ins.op == Op.MOV:
        return f"{s(ins.dst)} = mov {s(ins.args[0])}"
    if ins.op == Op.BIN:
        return f"{s(ins.dst)} = {BinOp(ins.sub).name.lower()} {s(ins.args[0])}, {s(ins.args[1])}"
    if ins.op == Op.UN:
        return f"{s(ins.dst)} = {UnOp(ins.sub).name.lower()} {s(ins.args[0])}"
    if ins.op == Op.CALL:
        callee = module.functions[ins.imm].name if 0 <= ins.imm < len(module.functions) else f"fn{ins.imm}"
        args = ", ".join(s(a) for a in ins.args)
        return (f"{s(ins.dst)} = " if ins.dst is not None else "") + f"call {callee}({args})"
    if ins.op == Op.INTR:
        name = Intrinsic(ins.sub).name.lower()
        args = ", ".join(s(a) for a in ins.args)
        if ins.sub == Intrinsic.WRITE_STR:
            args = f"str{ins.imm}"
        return (f"{s(ins.dst)} = " if ins.dst is not None else "") + f"{name}({args})"
    if ins.op == Op.JMP:
        return f"jmp b{ins.targets[0]}"
    if ins.op == Op.BR:
        return f"br {s(ins.args[0])} ? b{ins.targets[0]} : b{ins.targets[1]}"
    if ins.op == Op.RET:
        return "ret" + (f" {s(ins.args[0])}" if ins.args else "")
    return f"?? {ins}"
