"""Command-line interface.

    python -m aimc run "print the first 10 fibonacci numbers"
    python -m aimc compile "print hello world" -o hello --target aarch64-macos
    python -m aimc ir "print 6 times 7" --text
    python -m aimc verify "read a number and print it doubled" --stdin 21 --expect-stdout "42"
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import List, Optional

from . import __version__
from .codegen import SUPPORTED_TARGETS, Target, compile_module, host_target
from .ir import Interpreter, disassemble, from_json, to_json, verify as verify_ir
from .runtime import Sandbox, UnsupportedTarget, executable_suffix, write_executable


def _read_source(arg: str) -> str:
    if arg == "-":
        return sys.stdin.read()
    if os.path.exists(arg):
        with open(arg, "r", encoding="utf-8") as fh:
            return fh.read()
    return arg


def _load_module(arg: str, is_ir: bool, frontend: str = "rules"):
    text = _read_source(arg)
    if is_ir or text.lstrip().startswith("{"):
        m = from_json(text)
        verify_ir(m)
        return m
    if frontend == "claude":
        from .intent.llm import parse_with_model
        return parse_with_model(text)
    from .intent import parse
    return parse(text)


def _stdin_bytes(args) -> bytes:
    if getattr(args, "stdin_file", None):
        with open(args.stdin_file, "rb") as fh:
            return fh.read()
    if getattr(args, "stdin", None) is not None:
        s = args.stdin
        return (s if s.endswith("\n") else s + "\n").encode()
    return b""


def cmd_ir(args) -> int:
    m = _load_module(args.source, args.ir, args.frontend)
    if args.text:
        sys.stdout.write(disassemble(m))
    else:
        sys.stdout.write(to_json(m, pretty=not args.compact) + "\n")
    return 0


def cmd_explain(args) -> int:
    from .intent import explain
    print(explain(_read_source(args.source)))
    return 0


def cmd_compile(args) -> int:
    m = _load_module(args.source, args.ir, args.frontend)
    target = Target.parse(args.target) if args.target else host_target()
    image = compile_module(m, target)
    if args.format == "exe":
        data = write_executable(image)
    elif args.format == "raw":
        data = image.code
    else:
        data = (image.code.hex() + "\n").encode()
    out = args.output or ("a.out" if args.format == "exe" else "a." + args.format)
    with open(out, "wb") as fh:
        fh.write(data)
    if args.format == "exe":
        os.chmod(out, 0o755)
    print(f"{out}: {target} {args.format}, {len(image.code)} bytes of code, "
          f"_start=+{image.start_offset:#x} aimc_entry=+{image.entry_offset:#x}", file=sys.stderr)
    return 0


def cmd_run(args) -> int:
    m = _load_module(args.source, args.ir, args.frontend)
    stdin = _stdin_bytes(args)
    if args.strategy == "interp":
        r = Interpreter(m, stdin=stdin).run()
        sys.stdout.buffer.write(r.stdout)
        sys.stderr.buffer.write(r.stderr)
        return r.exit_code
    target = Target.parse(args.target) if args.target else host_target()
    image = compile_module(m, target)
    try:
        res = Sandbox(timeout=args.timeout).run(image, stdin=stdin, strategy=args.strategy)
    except UnsupportedTarget as e:
        print(f"aimc: {e}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(res.stdout)
    sys.stdout.flush()
    sys.stderr.buffer.write(res.stderr)
    if res.timed_out:
        print("aimc: program timed out", file=sys.stderr)
        return 124
    if res.signal is not None:
        print(f"aimc: program killed by signal {res.signal}", file=sys.stderr)
        return 128 + res.signal
    return res.exit_code


def cmd_verify(args) -> int:
    from .verify import verify_module, verify_prompt
    stdin = _stdin_bytes(args)
    exp_out = None
    if args.expect_stdout is not None:
        exp_out = args.expect_stdout.encode().replace(b"\\n", b"\n")
        if not exp_out.endswith(b"\n") and not args.exact:
            exp_out += b"\n"
    targets = [Target.parse(t) for t in args.target] if args.target else None
    strategies = tuple(args.strategy) if args.strategy else ("exe",)
    if args.ir or _read_source(args.source).lstrip().startswith("{"):
        report = verify_module(_load_module(args.source, True), stdin, exp_out, args.expect_exit, targets, strategies)
    else:
        report = verify_prompt(_read_source(args.source), stdin, exp_out, args.expect_exit, targets, strategies)
    print(report.summary())
    return 0 if report.passed else 1


def cmd_targets(args) -> int:
    sb = Sandbox()
    host = host_target()
    for name in SUPPORTED_TARGETS:
        t = Target.parse(name)
        strategies = sb.strategies(t)
        tag = " (host)" if t == host else ""
        print(f"{name:16} runnable via: {', '.join(strategies) or 'none'}{tag}")
    return 0


def cmd_disasm(args) -> int:
    """Disassemble the generated code by writing the real executable and handing it to objdump."""
    import re
    import tempfile
    from .runtime import code_vaddr
    m = _load_module(args.source, args.ir, args.frontend)
    target = Target.parse(args.target) if args.target else host_target()
    image = compile_module(m, target)
    tool = shutil.which("llvm-objdump") or shutil.which("objdump")
    if not tool:
        print("no objdump available; dumping hex", file=sys.stderr)
        print(image.code.hex())
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "program" + executable_suffix(image))
        with open(path, "wb") as fh:
            fh.write(write_executable(image))
        r = subprocess.run([tool, "-d", path], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return 1
    base = code_vaddr(image)
    by_off = {off: name for name, off in image.symbols.items() if not name.startswith(".")}
    addr_re = re.compile(r"^\s*([0-9a-fA-F]+):")
    printed_any = False
    for line in r.stdout.splitlines():
        mm = addr_re.match(line)
        if mm:
            off = int(mm.group(1), 16) - base
            if off in by_off:
                print(f"\n{by_off[off]}:  (+{off:#x})")
            if off >= image.rodata_offset and printed_any:
                continue  # data bytes disassembled as garbage: stop at rodata
            printed_any = True
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aimc", description="intent -> machine code compiler")
    p.add_argument("--version", action="version", version=f"aimc {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def source_args(sp, with_target=True):
        sp.add_argument("source", help="prompt text, a file containing it, an IR JSON file, or '-' for stdin")
        sp.add_argument("--ir", action="store_true", help="treat the source as IR JSON")
        sp.add_argument("--frontend", choices=["rules", "claude"], default="rules",
                        help="rules: built-in semantic parser (default); claude: ask a Claude model to emit IR (needs `anthropic`)")
        if with_target:
            sp.add_argument("-t", "--target", help=f"one of {', '.join(SUPPORTED_TARGETS)} (default: host)")

    sp = sub.add_parser("ir", help="show the IR for a prompt")
    source_args(sp, with_target=False)
    sp.add_argument("--text", action="store_true", help="human-oriented disassembly instead of JSON")
    sp.add_argument("--compact", action="store_true", help="canonical single-line JSON")
    sp.set_defaults(func=cmd_ir)

    sp = sub.add_parser("explain", help="describe how a prompt was understood")
    sp.add_argument("source")
    sp.set_defaults(func=cmd_explain)

    sp = sub.add_parser("compile", help="compile to an executable / raw code")
    source_args(sp)
    sp.add_argument("-o", "--output")
    sp.add_argument("-f", "--format", choices=["exe", "raw", "hex"], default="exe")
    sp.set_defaults(func=cmd_compile)

    sp = sub.add_parser("run", help="compile and run in the sandbox")
    source_args(sp)
    sp.add_argument("--stdin", help="text fed to the program's standard input")
    sp.add_argument("--stdin-file")
    sp.add_argument("--strategy", choices=["auto", "exe", "jit", "interp"], default="auto")
    sp.add_argument("--timeout", type=float, default=5.0)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("verify", help="differentially verify interpreter vs. native runs")
    sp.add_argument("source")
    sp.add_argument("--ir", action="store_true")
    sp.add_argument("-t", "--target", action="append", help="repeatable; default: every runnable target")
    sp.add_argument("--strategy", action="append", choices=["exe", "jit"])
    sp.add_argument("--stdin")
    sp.add_argument("--stdin-file")
    sp.add_argument("--expect-stdout", help="expected output ('\\n' escapes allowed; trailing newline implied)")
    sp.add_argument("--expect-exit", type=int)
    sp.add_argument("--exact", action="store_true", help="do not imply a trailing newline on --expect-stdout")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("targets", help="list targets and how they can run here")
    sp.set_defaults(func=cmd_targets)

    sp = sub.add_parser("disasm", help="disassemble generated code (needs llvm-objdump/objdump)")
    source_args(sp)
    sp.set_defaults(func=cmd_disasm)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except UnsupportedTarget as e:
        print(f"aimc: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        from .intent.llm import ModelFrontendError
        if isinstance(e, ModelFrontendError):
            print(f"aimc: model frontend failed: {e}", file=sys.stderr)
            return 3
        from .intent import IntentError
        if isinstance(e, IntentError):
            print(f"aimc: could not understand the request: {e}", file=sys.stderr)
            return 3
        raise
