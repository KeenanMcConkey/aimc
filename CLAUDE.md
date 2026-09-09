# aimc — Intent-to-Machine-Code Compiler

`aimc` compiles natural-language program specifications straight to raw
x86_64 / AArch64 machine code. There is no C, no assembler, no linker and no
libc in the loop: the intent parser produces a machine-oriented IR, the
backends encode instruction bytes directly, and the runtime wraps those bytes
in an ELF or (self-signed) Mach-O container or maps them into memory and jumps
to them.

```
 "print the first 10 fibonacci numbers"
          │  aimc.intent      rule-based semantic parser (tokens → patterns → IR)
          ▼
   IR module (aimc.ir)        slots + basic blocks; verified; canonical JSON + digest
          │  aimc.ir.interp   reference interpreter = executable semantics
          │  aimc.codegen     x86_64 / aarch64 encoders emit position-independent bytes
          ▼
   code image                 [_start][aimc_entry][fn*][runtime routines][rodata]
          │  aimc.runtime     ELF64 / Mach-O writer (+ ad-hoc CodeDirectory signature), JIT loader
          ▼
   sandboxed process          stdin/stdout/exit code captured; compared by aimc.verify
```

Pure Python 3.9+, standard library only. Developed and tested on Apple Silicon
(native arm64 + Rosetta x86_64); Linux targets are produced as static ELF
binaries and can be run on a Linux host.

## Running

```bash
cd aimc                                    # this directory
python3 -m aimc targets                    # which targets can execute on this machine
python3 -m aimc run "print hello world"
python3 -m aimc run "read two numbers and print their sum" --stdin "40 2"
python3 -m aimc run examples/fizzbuzz.txt --target x86_64-macos --strategy jit
python3 -m aimc ir "print 6 times 7" --text          # IR disassembly
python3 -m aimc ir "print 6 times 7"                 # canonical JSON (the verification artifact)
python3 -m aimc explain "for each number from 1 to 5, print it squared"
python3 -m aimc compile "print hello world" -o hello --target aarch64-macos && ./hello
python3 -m aimc compile "print hello world" -o hello.elf --target x86_64-linux   # run on Linux
python3 -m aimc compile "print 42" -f raw -o code.bin                            # bare code bytes
python3 -m aimc disasm "print 42" --target aarch64-macos                         # needs objdump
python3 -m aimc verify "read a number and print it doubled" --stdin 21 --expect-stdout 42
python3 -m aimc run '{"aimc_ir": 1, ...}' --ir     # run hand-written / model-written IR JSON
python3 -m aimc run "print the first 10 primes" --frontend claude   # let Claude write the IR (needs `pip install anthropic` + credentials)
```

Targets: `x86_64-linux`, `x86_64-macos`, `aarch64-linux`, `aarch64-macos`
(default: the host). Execution strategies: `exe` (write a real executable and
spawn it), `jit` (child Python process maps the bytes and calls `aimc_entry`),
`interp` (reference interpreter only). `pip install -e .` installs an `aimc`
console script if you prefer that to `python3 -m aimc`.

## Testing

```bash
python3 -m unittest discover -s tests -t . -v     # everything (~45 s on an M-series Mac)
python3 -m unittest tests.test_intent             # parser: prompts → IR → interpreter
python3 -m unittest tests.test_codegen            # encoders + backend vs interpreter
python3 -m unittest tests.test_runtime            # ELF/Mach-O/signature/sandbox
python3 -m unittest tests.test_encoders_vs_clang  # byte-exact cross-check vs clang (skips without clang)
python3 -m unittest tests.test_end_to_end         # English prompt → native binary → asserted I/O
python3 -m unittest tests.test_cli tests.test_llm_frontend
```

The end-to-end suite is the acceptance test: every prompt is parsed, the IR
is verified and round-tripped, executed in the interpreter, compiled for every
target, executed natively on every target the host can run (both `exe` and
`jit`), and all outputs must agree with each other and with the expected
stdout / exit status. Tests for targets the host cannot execute are skipped,
not faked.

## The IR (what an automated reasoner sees)

Full semantics are documented at the top of `aimc/ir/nodes.py`; `aimc.ir.interp`
is the executable specification every backend must match, and `aimc.ir.verify`
rejects malformed graphs and any slot read before it is written on some path.

## What the intent parser understands

See `aimc/docs/prompt-grammar.md` for the supported sentence and expression
forms. Unrecognised clauses raise `IntentError`; nothing is guessed silently.

## Extending

**New intent / sentence shape.** `aimc/intent/statements.py` holds a table of
`(regex, handler)` patterns; add an entry that builds a statement node, and
lowering in `aimc/intent/lower.py` turns nodes into IR. Expression forms live
in `aimc/intent/expressions.py`. Add a test to `tests/test_intent.py` (runs in
the interpreter) and a prompt to `tests/test_end_to_end.py` (runs natively).

**New IR helper (e.g. `is_perfect(n)`).** Add a builder function to
`aimc/intent/stdlib.py`; helpers are IR functions so every backend gets them
for free.

**New intrinsic / opcode.** Add it to `aimc/ir/nodes.py` (enum + arity
table), the interpreter, the verifier, and each backend's `_intr` / `_bin`
plus its `rt.*` routine if it needs machine-level support. Semantics must be
identical in the interpreter and both backends; the differential tests will
tell you if they are not.

**New target.** Add a syscall table entry in `aimc/codegen/target.py`; for a
new OS on an existing architecture that is usually all a backend needs
(check the error-reporting convention of `read`/`write` — macOS uses the
carry flag). A new architecture needs an `encoder.py` (subclass
`aimc.codegen.image.Assembler`, implement `patch` for its fixup kinds) and a
`backend.py` that emits the same image layout and `rt.*` routines; add a
container writer in `aimc/runtime/` if the OS needs one.

**Different frontend.** Anything that produces an IR module can drive the
rest of the pipeline: `python3 -m aimc run file.json --ir` accepts IR JSON,
and `aimc.intent.llm` shows how a language model can be asked to emit IR that
is then verified before it is executed.

## Design constraints worth knowing

* Backends allocate every IR slot a stack word and load/compute/store per
  instruction. It is deliberately simple and uniformly verifiable; a register
  allocator would be a peephole layer above this, not a change to the IR.
* Generated code only touches caller-saved registers, so `aimc_entry` is a
  well-behaved C function — that is what makes the in-process JIT path safe.
* macOS on Apple Silicon refuses static executables and unsigned code, so
  Mach-O images are dyld-linked (`LC_MAIN`, no libraries) and carry an
  ad-hoc SHA-256 CodeDirectory generated in `aimc/runtime/macho.py`.
* The JIT loader must stay Python 3.8-compatible and stdlib-only: on Apple
  Silicon the arm64 slice is often executed by `/usr/bin/python3`.
