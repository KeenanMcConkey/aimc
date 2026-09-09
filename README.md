# aimc

**An intent-to-machine-code compiler.** You describe a program in English;
`aimc` emits raw x86_64 or AArch64 machine code for it. There is no C, no
assembler, no linker and no libc anywhere in the pipeline: the intent parser
produces a small machine-oriented IR, the backends encode instruction bytes
directly, and the runtime wraps those bytes in an ELF or self-signed Mach-O
executable, or maps them into memory and jumps to them.

```
$ python3 -m aimc run "print fizzbuzz up to 15"
1
2
Fizz
4
Buzz
...
FizzBuzz

$ python3 -m aimc run "read two numbers and print their sum" --stdin "40 2"
42

$ python3 -m aimc compile "print hello world" -o hello && ./hello
hello world
```

That `hello` is a 16 KB Mach-O (or ELF) whose only contents are 633 bytes of
generated code and a code signature. Pure Python 3.9+, standard library only.

## How it works

```
 "read a number and print it doubled"
          │  aimc.intent      rule-based semantic parser: tokens → patterns → IR
          ▼
   IR module (aimc.ir)        i64 slots + basic blocks; verified; canonical JSON + SHA-256 digest
          │  aimc.ir.interp   reference interpreter = the executable semantics
          │  aimc.codegen     x86_64 / aarch64 encoders emit position-independent bytes
          ▼
   code image                 [_start][aimc_entry][fn*][runtime routines][rodata]
          │  aimc.runtime     ELF64 / Mach-O writer (+ ad-hoc CodeDirectory signature), JIT loader
          ▼
   sandboxed process          stdin/stdout/exit status captured; cross-checked by aimc.verify
```

The IR is deliberately not for humans. Every value is a 64-bit integer in a
numbered slot, every instruction does one machine-sized thing, and control
flow is an explicit block graph with single terminators:

```
$ python3 -m aimc ir "read a number and print it doubled" --text
; aimc IR v1  digest=6a306fdb4c0eabd6
fn0 main(params=0, slots=5) (entry):
  b0:
    s0 = const 0
    jmp b1
  b1:
    s0 = read_int()
    s1 = const 2
    s2 = mul s0, s1
    write_int(s2)
    s3 = const 10
    write_char(s3)
    s4 = const 0
    ret s4
```

A verifier checks the graph, arities, and definite assignment (no slot read
before it is written on any path). The interpreter defines the semantics
bit-for-bit (wrapping arithmetic, truncating division, a defined trap on
division by zero), and every backend is held to it.

## Verification is the point

`aimc verify` runs a prompt through every stage and demands agreement:

```
$ python3 -m aimc verify "read a number and print it doubled" --stdin 21 --expect-stdout 42 --strategy exe --strategy jit
verification of 'read a number and print it doubled': PASS
  ir digest: 6a306fdb4c0eabd6400c445192f2893782225c132afa246001ff90177520d226
  [ok  ] intent: 1 function(s), 10 instructions
  [ok  ] ir-verify
  [ok  ] ir-roundtrip
  [ok  ] interpret: exit=0 stdout=b'42\n' steps=10
  [ok  ] codegen[x86_64-macos]: 685 bytes
  [ok  ] run[x86_64-macos/exe]: matches interpreter
  [ok  ] run[x86_64-macos/jit]: matches interpreter
  [ok  ] codegen[aarch64-macos]: 669 bytes
  [ok  ] run[aarch64-macos/exe]: matches interpreter
  [ok  ] run[aarch64-macos/jit]: matches interpreter
```

Two execution strategies exist so they can check each other: `exe` writes a
real executable and spawns it; `jit` maps the raw bytes into a child Python
process and calls the C-ABI entry point. Both run in a sandbox with a timeout,
resource limits, and captured I/O.

## Targets

| Target | Container | Notes |
|---|---|---|
| `aarch64-macos` | Mach-O, dyld-linked, ad-hoc signed | Apple Silicon refuses static and unsigned arm64 code; the signature is generated in pure Python |
| `x86_64-macos` | Mach-O, ad-hoc signed | runs under Rosetta on Apple Silicon |
| `x86_64-linux` | static ELF64 | |
| `aarch64-linux` | static ELF64 | |

All four are always produced; `python3 -m aimc targets` shows which ones the
current machine can execute.

## Install and run

```bash
git clone https://github.com/KeenanMcConkey/aimc && cd aimc
python3 -m aimc run "print the first 10 fibonacci numbers"     # no dependencies
pip install -e .                                                # optional: gives you an `aimc` command
```

Commands: `run`, `compile` (`-f exe|raw|hex`), `ir` (`--text` or JSON),
`explain`, `verify`, `disasm` (needs objdump), `targets`. Any command accepts
a prompt string, a file containing one, IR JSON (`--ir`), or `-` for stdin.

## What it understands

The default frontend is a deterministic, table-driven semantic parser: no
network, no model. It handles printing, arithmetic in infix or English
("the remainder when x is divided by 3", "n squared", "the gcd of a and b"),
variables and updates, reading integers from stdin, `if`/`else`, `while`/
`until`, counted loops, exit codes, and a set of idioms (fizzbuzz, fibonacci,
primes, collatz, countdowns, multiplication tables, ...). Unrecognised
clauses raise an error naming the clause; nothing is guessed silently. The
full grammar is in [`docs/prompt-grammar.md`](docs/prompt-grammar.md).

There is also an optional model-backed frontend: `--frontend claude` asks a
Claude model to emit the IR JSON directly (still no source code in any
language), and the same verifier and differential checks decide whether the
result is acceptable, feeding verifier errors back for repair. It needs
`pip install anthropic` and credentials; the tests exercise it with a stub.

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

114 tests: IR semantics, byte-exact encoder checks (cross-checked against
clang's assembler when available), differential backend tests over an
operand grid, container/signature tests, sandbox tests, 59 parser tests, and
an end-to-end suite of 33 English prompts executed natively on every
runnable target via both strategies.

## Layout

```
aimc/intent/    natural language → IR   (lexer, expressions, statements, lower, stdlib, llm)
aimc/ir/        nodes, builder, verify, serialize, interp
aimc/codegen/   target table, assembler core, x86_64/ and aarch64/ encoders + backends
aimc/runtime/   elf, macho (+ signer), JIT loader, sandbox
aimc/verify/    differential verifier with per-stage reports
tests/          the suites above; tests/support.py is the harness
examples/       prompt files
```

`CLAUDE.md` documents how to extend each layer (new sentence shapes, new
intrinsics, new targets).

## Status and limitations

Early but complete end to end. Backends are -O0 style: every IR slot lives in
a stack word and each instruction loads, computes, and stores, which keeps
codegen a uniformly verifiable table walk. The parser is a bounded grammar,
not a general language model; integers only, no strings as values, no
user-defined functions yet. Linux binaries are produced but were only
structurally checked on the development machine.

## License

MIT
