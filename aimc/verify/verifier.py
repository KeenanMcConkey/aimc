"""The verifier ties the pipeline together and checks every stage against the others.

For a prompt (or an IR module) it:
  1. parses the intent into IR and runs the structural verifier
  2. checks the IR round-trips through canonical JSON with an identical digest
  3. executes the IR in the reference interpreter
  4. compiles the IR for each requested target and runs the raw machine code
     (native executable and/or JIT loader) in the sandbox
  5. asserts that every native run reproduces the interpreter's stdout, stderr
     and exit status byte-for-byte, and -- when given -- the caller's expectations

The result is a Report whose checks are individually inspectable, so a test
suite (or an agent) can see exactly which stage disagreed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from aimc.codegen import Image, SUPPORTED_TARGETS, Target, compile_module, host_target
from aimc.ir import Interpreter, Module, digest, from_json, to_json, verify as verify_ir
from aimc.ir.interp import ExecResult
from aimc.runtime import RunResult, Sandbox


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        mark = "ok  " if self.passed else "FAIL"
        return f"[{mark}] {self.name}" + (f": {self.detail}" if self.detail else "")


@dataclass
class Report:
    source: str
    module: Optional[Module] = None
    ir_digest: str = ""
    reference: Optional[ExecResult] = None
    images: dict = field(default_factory=dict)      # target name -> Image
    runs: dict = field(default_factory=dict)        # (target, strategy) -> RunResult
    checks: List[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str = "") -> Check:
        c = Check(name, passed, detail)
        self.checks.append(c)
        return c

    def summary(self) -> str:
        lines = [f"verification of {self.source!r}: {'PASS' if self.passed else 'FAIL'}"]
        if self.ir_digest:
            lines.append(f"  ir digest: {self.ir_digest}")
        lines.extend("  " + str(c) for c in self.checks)
        return "\n".join(lines)


def runnable_targets(sandbox: Optional[Sandbox] = None) -> List[Target]:
    sb = sandbox or Sandbox()
    return [Target.parse(t) for t in SUPPORTED_TARGETS if sb.strategies(Target.parse(t))]


def verify_prompt(prompt: str, stdin: bytes = b"", expected_stdout: Optional[bytes] = None,
                  expected_exit: Optional[int] = None, targets: Optional[Sequence[Target]] = None,
                  strategies: Iterable[str] = ("exe",), sandbox: Optional[Sandbox] = None) -> Report:
    from aimc.intent import IntentError, parse
    report = Report(prompt)
    try:
        module = parse(prompt)
    except IntentError as e:
        report.add("intent", False, str(e))
        return report
    report.add("intent", True, f"{len(module.functions)} function(s), {sum(len(b.instrs) for f in module.functions for b in f.blocks)} instructions")
    return verify_module(module, stdin, expected_stdout, expected_exit, targets, strategies, sandbox, report)


def verify_module(module: Module, stdin: bytes = b"", expected_stdout: Optional[bytes] = None,
                  expected_exit: Optional[int] = None, targets: Optional[Sequence[Target]] = None,
                  strategies: Iterable[str] = ("exe",), sandbox: Optional[Sandbox] = None,
                  report: Optional[Report] = None) -> Report:
    report = report or Report("<module>")
    report.module = module
    sb = sandbox or Sandbox()

    # 1. structural verification
    try:
        verify_ir(module)
        report.add("ir-verify", True)
    except Exception as e:  # IRError
        report.add("ir-verify", False, str(e))
        return report

    # 2. canonical form round-trip
    report.ir_digest = digest(module)
    rt = from_json(to_json(module))
    report.add("ir-roundtrip", digest(rt) == report.ir_digest)

    # 3. reference semantics
    try:
        ref = Interpreter(module, stdin=stdin).run()
    except Exception as e:
        report.add("interpret", False, f"{type(e).__name__}: {e}")
        return report
    report.reference = ref
    report.add("interpret", True, f"exit={ref.exit_code} stdout={_short(ref.stdout)} steps={ref.steps}")
    if expected_stdout is not None:
        report.add("interp-expected-stdout", ref.stdout == expected_stdout, _diff(expected_stdout, ref.stdout))
    if expected_exit is not None:
        report.add("interp-expected-exit", ref.exit_code == expected_exit, f"expected {expected_exit}, got {ref.exit_code}")

    # 4./5. native execution on each target
    tlist = list(targets) if targets is not None else runnable_targets(sb) or [host_target()]
    for target in tlist:
        try:
            image = compile_module(module, target)
        except Exception as e:
            report.add(f"codegen[{target}]", False, f"{type(e).__name__}: {e}")
            continue
        report.images[target.name] = image
        report.add(f"codegen[{target}]", True, f"{len(image.code)} bytes")
        avail = sb.strategies(target)
        for strategy in strategies:
            if strategy not in avail:
                report.add(f"run[{target}/{strategy}]", True, "skipped: not executable on this host")
                continue
            res = sb.run(image, stdin=stdin, strategy=strategy)
            report.runs[(target.name, strategy)] = res
            name = f"run[{target}/{strategy}]"
            if res.timed_out:
                report.add(name, False, "timed out")
                continue
            if res.signal is not None:
                report.add(name, False, f"killed by signal {res.signal}")
                continue
            same = (res.stdout, res.stderr, res.exit_code) == (ref.stdout, ref.stderr, ref.exit_code)
            report.add(name, same, "matches interpreter" if same else
                       f"exit {res.exit_code} vs {ref.exit_code}; stdout {_diff(ref.stdout, res.stdout)}; stderr {_diff(ref.stderr, res.stderr)}")
            if expected_stdout is not None:
                report.add(name + "-expected-stdout", res.stdout == expected_stdout, _diff(expected_stdout, res.stdout))
            if expected_exit is not None:
                report.add(name + "-expected-exit", res.exit_code == expected_exit, f"expected {expected_exit}, got {res.exit_code}")
    return report


def _short(b: bytes, n: int = 60) -> str:
    s = repr(b)
    return s if len(s) <= n else s[:n] + "..."


def _diff(expected: bytes, got: bytes) -> str:
    if expected == got:
        return "ok"
    return f"expected {_short(expected)}, got {_short(got)}"
