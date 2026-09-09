"""Optional model-backed frontend: ask Claude to emit IR JSON directly.

The rule-based parser in this package is the default frontend.  This module
shows the alternative the IR was designed for: a language model writes the
IR *itself* (no source code in any human language is generated), and the
compiler's own verifier, interpreter and differential checks decide whether
the result is acceptable.  Verification failures are fed back to the model
for a bounded number of repair rounds.

    from aimc.intent.llm import parse_with_model
    module = parse_with_model("print the first 10 primes")   # needs the `anthropic` package + credentials

No network access happens at import time, and `client` can be any object
with a `.beta.messages.create(...)` method (the tests use a stub).
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional

from aimc.ir import IRError, Module, from_json, verify
from aimc.ir.nodes import BinOp, Intrinsic, Op, UnOp

DEFAULT_MODEL = "claude-opus-5"


class ModelFrontendError(Exception):
    pass


def _opcode_table() -> str:
    lines = ["Opcodes (op field):"]
    lines += [f"  {int(o)} = {o.name}" for o in Op]
    lines.append("BinOp codes (sub field when op=BIN):")
    lines += [f"  {int(o)} = {o.name}" for o in BinOp]
    lines.append("UnOp codes (sub field when op=UN):")
    lines += [f"  {int(o)} = {o.name}" for o in UnOp]
    lines.append("Intrinsic codes (sub field when op=INTR):")
    lines += [f"  {int(o)} = {o.name}" for o in Intrinsic]
    return "\n".join(lines)


IR_SPEC = f"""You are the frontend of a compiler. Translate the user's program description into
aimc IR, a tiny machine-oriented intermediate representation, and reply with ONLY a
single JSON object (optionally inside a ```json fence). Do not write source code in any
programming language; write the IR directly.

JSON shape:
{{"aimc_ir": 1, "entry": <index of the entry function>,
 "strings": ["text constants, \\n for newline"],
 "functions": [{{"name": "main", "params": 0, "slots": <number of slots>,
               "blocks": [[<instr>, ...], ...]}}]}}

Every value is a signed 64-bit integer held in a numbered slot (0-based, local to the
function; parameters occupy slots 0..params-1). An instruction is a 6-element list
[op, sub, dst, args, imm, targets] where dst is a slot or null, args is a list of slots,
imm is an integer and targets is a list of block indices.

{_opcode_table()}

Instruction forms:
  CONST:  [1, 0, dst, [], value, []]                 dst = value
  MOV:    [2, 0, dst, [src], 0, []]                  dst = src
  BIN:    [3, binop, dst, [a, b], 0, []]             dst = a <binop> b   (comparisons give 0/1)
  UN:     [4, unop, dst, [a], 0, []]                 dst = <unop> a      (NOT is logical)
  CALL:   [5, 0, dst_or_null, [args...], fn_index, []]
  INTR:   [6, 0, null, [x], 0, []]                   WRITE_INT: print decimal text of x (no newline)
          [6, 1, null, [], string_index, []]         WRITE_STR: print strings[string_index]
          [6, 2, null, [x], 0, []]                   WRITE_CHAR: print the byte x (10 = newline)
          [6, 3, dst, [], 0, []]                     READ_INT: dst = next integer from stdin
          [6, 4, null, [code], 0, []]                EXIT: terminate with exit status code
  JMP:    [16, 0, null, [], 0, [block]]
  BR:     [17, 0, null, [cond], 0, [if_true_block, if_false_block]]
  RET:    [18, 0, null, [value], 0, []]              (args may be [] to return 0)

Rules: block 0 is the function entry; every block ends with exactly one terminator
(JMP/BR/RET) and has no terminator earlier; every slot must be written on all paths
before it is read; arithmetic wraps at 64 bits; DIV truncates toward zero; DIV/REM by
zero traps. The entry function takes 0 params and its return value (low 8 bits) is the
process exit status. "print X" means WRITE_INT then WRITE_CHAR 10 (or WRITE_STR of the
text with a trailing newline). Prefer loops over unrolling. Keep it minimal.
"""

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def extract_json(text: str) -> str:
    m = _FENCE.search(text)
    if m:
        return m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ModelFrontendError("model reply contained no JSON object")
    return text[start:end + 1]


def module_from_reply(text: str) -> Module:
    try:
        module = from_json(extract_json(text))
    except (ValueError, KeyError, TypeError) as e:
        raise ModelFrontendError(f"reply is not valid aimc IR JSON: {e}")
    verify(module)
    return module


def parse_with_model(text: str, model: str = DEFAULT_MODEL, client: Any = None,
                     max_repairs: int = 2, effort: str = "high") -> Module:
    """Ask a model for IR, verify it, and feed verifier errors back for repair."""
    if client is None:
        try:
            import anthropic
        except ImportError:
            raise ModelFrontendError("the `anthropic` package is required for the model frontend (pip install anthropic)")
        client = anthropic.Anthropic()

    messages: List[dict] = [{"role": "user", "content": text}]
    last_error = ""
    for attempt in range(max_repairs + 1):
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=IR_SPEC,
            messages=messages,
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise ModelFrontendError("the model declined to translate this request")
        reply = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")
        try:
            return module_from_reply(reply)
        except (ModelFrontendError, IRError) as e:
            last_error = str(e)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content":
                             f"The IR was rejected by the verifier: {last_error}\n"
                             f"Reply with a corrected complete JSON object only."})
    raise ModelFrontendError(f"model produced unverifiable IR after {max_repairs + 1} attempts: {last_error}")
