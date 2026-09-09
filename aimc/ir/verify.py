"""Structural and dataflow verification of IR modules.

Checks performed:
  * entry function exists and takes no parameters
  * block ids are dense and equal their position
  * every block is non-empty, ends in exactly one terminator, and has no
    terminator earlier in the block
  * every jump target, slot index, callee index, string index and intrinsic
    code is in range; call/intrinsic arities match
  * definite assignment: every slot is written on all paths before it is read
    (parameters are defined on entry)
"""
from __future__ import annotations

from typing import List, Set

from .nodes import (INTRINSIC_ARITY, INTRINSIC_RETURNS, BinOp, Function, Instr, Intrinsic,
                    Module, Op, UnOp)


class IRError(Exception):
    pass


def check(module: Module) -> List[str]:
    errors: List[str] = []
    if not module.functions:
        return ["module has no functions"]
    if not 0 <= module.entry < len(module.functions):
        errors.append(f"entry index {module.entry} out of range")
    else:
        ef = module.functions[module.entry]
        if ef.nparams != 0:
            errors.append(f"entry function {ef.name!r} must take 0 parameters, takes {ef.nparams}")
    names = [f.name for f in module.functions]
    if len(set(names)) != len(names):
        errors.append("duplicate function names")
    for f in module.functions:
        errors.extend(_check_function(module, f))
    return errors


def verify(module: Module) -> None:
    errors = check(module)
    if errors:
        raise IRError("; ".join(errors))


def _check_function(module: Module, f: Function) -> List[str]:
    e: List[str] = []
    where = f"function {f.name!r}"
    if not f.blocks:
        return [f"{where}: no blocks"]
    if f.nparams > f.nslots:
        e.append(f"{where}: nparams {f.nparams} exceeds nslots {f.nslots}")
    nblocks = len(f.blocks)
    for pos, b in enumerate(f.blocks):
        bw = f"{where} block {b.id}"
        if b.id != pos:
            e.append(f"{bw}: id does not match position {pos}")
        if not b.instrs:
            e.append(f"{bw}: empty block")
            continue
        for i, ins in enumerate(b.instrs):
            last = i == len(b.instrs) - 1
            if ins.is_terminator != last:
                e.append(f"{bw} instr {i}: terminator placement")
            e.extend(_check_instr(module, f, ins, nblocks, f"{bw} instr {i}"))
    if not e:
        e.extend(_check_definite_assignment(f))
    return e


def _slot_ok(f: Function, s) -> bool:
    return isinstance(s, int) and 0 <= s < f.nslots


def _check_instr(module: Module, f: Function, ins: Instr, nblocks: int, w: str) -> List[str]:
    e: List[str] = []
    for a in ins.args:
        if not _slot_ok(f, a):
            e.append(f"{w}: slot {a} out of range")
    for t in ins.targets:
        if not 0 <= t < nblocks:
            e.append(f"{w}: block target {t} out of range")
    if ins.dst is not None and not _slot_ok(f, ins.dst):
        e.append(f"{w}: dst slot {ins.dst} out of range")

    def need(nargs: int, ntargets: int, has_dst: bool) -> None:
        if len(ins.args) != nargs:
            e.append(f"{w}: expected {nargs} args, got {len(ins.args)}")
        if len(ins.targets) != ntargets:
            e.append(f"{w}: expected {ntargets} targets, got {len(ins.targets)}")
        if has_dst and ins.dst is None:
            e.append(f"{w}: missing dst")
        if not has_dst and ins.dst is not None:
            e.append(f"{w}: unexpected dst")

    if ins.op == Op.CONST:
        need(0, 0, True)
        if not -(1 << 63) <= ins.imm < (1 << 64):
            e.append(f"{w}: immediate does not fit in 64 bits")
    elif ins.op == Op.MOV:
        need(1, 0, True)
    elif ins.op == Op.BIN:
        need(2, 0, True)
        if ins.sub not in BinOp.__members__.values():
            e.append(f"{w}: unknown binop {ins.sub}")
    elif ins.op == Op.UN:
        need(1, 0, True)
        if ins.sub not in UnOp.__members__.values():
            e.append(f"{w}: unknown unop {ins.sub}")
    elif ins.op == Op.CALL:
        if len(ins.targets):
            e.append(f"{w}: call has targets")
        if not 0 <= ins.imm < len(module.functions):
            e.append(f"{w}: callee index {ins.imm} out of range")
        else:
            callee = module.functions[ins.imm]
            if callee.nparams != len(ins.args):
                e.append(f"{w}: {callee.name} takes {callee.nparams} args, got {len(ins.args)}")
            if len(ins.args) > 6:
                e.append(f"{w}: more than 6 call arguments are not supported")
    elif ins.op == Op.INTR:
        if ins.sub not in Intrinsic.__members__.values():
            e.append(f"{w}: unknown intrinsic {ins.sub}")
        else:
            which = Intrinsic(ins.sub)
            need(INTRINSIC_ARITY[which], 0, which in INTRINSIC_RETURNS)
            if which == Intrinsic.WRITE_STR and not 0 <= ins.imm < len(module.strings):
                e.append(f"{w}: string index {ins.imm} out of range")
    elif ins.op == Op.JMP:
        need(0, 1, False)
    elif ins.op == Op.BR:
        need(1, 2, False)
    elif ins.op == Op.RET:
        if len(ins.args) > 1 or ins.targets or ins.dst is not None:
            e.append(f"{w}: malformed ret")
    else:
        e.append(f"{w}: unknown opcode {ins.op}")
    return e


def _check_definite_assignment(f: Function) -> List[str]:
    """Forward must-analysis: slot is defined on entry to a block iff defined on exit of every predecessor."""
    all_slots: Set[int] = set(range(f.nslots))
    params: Set[int] = set(range(f.nparams))
    preds = f.predecessors()
    out: List[Set[int]] = [set(all_slots) for _ in f.blocks]  # optimistic start
    reachable = _reachable(f)
    changed = True
    while changed:
        changed = False
        for b in f.blocks:
            if b.id not in reachable:
                continue
            if b.id == 0:
                cur = set(params)
            else:
                ps = [p for p in preds[b.id] if p in reachable]
                cur = set(all_slots)
                for p in ps:
                    cur &= out[p]
            for ins in b.instrs:
                if ins.dst is not None:
                    cur.add(ins.dst)
            if cur != out[b.id]:
                out[b.id] = cur
                changed = True
    errors: List[str] = []
    for b in f.blocks:
        if b.id not in reachable:
            continue
        cur = set(params) if b.id == 0 else set.intersection(
            set(all_slots), *[out[p] for p in preds[b.id] if p in reachable])
        for i, ins in enumerate(b.instrs):
            for a in ins.args:
                if a not in cur:
                    errors.append(f"function {f.name!r} block {b.id} instr {i}: slot {a} may be used before definition")
            if ins.dst is not None:
                cur.add(ins.dst)
    return errors


def _reachable(f: Function) -> Set[int]:
    seen: Set[int] = set()
    stack = [0]
    while stack:
        b = stack.pop()
        if b in seen or b >= len(f.blocks):
            continue
        seen.add(b)
        stack.extend(f.blocks[b].successors)
    return seen
