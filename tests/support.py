"""Shared helpers for the test suite."""
from __future__ import annotations

import unittest
from typing import List

from aimc.codegen import SUPPORTED_TARGETS, Target, compile_module
from aimc.ir import Interpreter, Module, verify
from aimc.runtime import Sandbox

SANDBOX = Sandbox(timeout=10.0)
ALL_TARGETS = [Target.parse(t) for t in SUPPORTED_TARGETS]


def runnable() -> List[tuple]:
    """(target, strategy) pairs executable on this machine."""
    return [(t, s) for t in ALL_TARGETS for s in SANDBOX.strategies(t)]


class PipelineCase(unittest.TestCase):
    """Base class: compile a module for every target and check native runs against the interpreter."""

    def check_module(self, module: Module, stdin: bytes = b"", expected_stdout=None, expected_exit=None,
                     expected_stderr=None):
        verify(module)
        ref = Interpreter(module, stdin=stdin).run()
        if expected_stdout is not None:
            self.assertEqual(ref.stdout, expected_stdout, "interpreter stdout")
        if expected_exit is not None:
            self.assertEqual(ref.exit_code, expected_exit, "interpreter exit code")
        if expected_stderr is not None:
            self.assertEqual(ref.stderr, expected_stderr, "interpreter stderr")
        images = {t.name: compile_module(module, t) for t in ALL_TARGETS}  # every backend must at least encode it
        ran = 0
        for target, strategy in runnable():
            with self.subTest(target=target.name, strategy=strategy):
                res = SANDBOX.run(images[target.name], stdin=stdin, strategy=strategy)
                self.assertFalse(res.timed_out, "timed out")
                self.assertIsNone(res.signal, f"killed by signal {res.signal}")
                self.assertEqual(res.stdout, ref.stdout)
                self.assertEqual(res.stderr, ref.stderr)
                self.assertEqual(res.exit_code, ref.exit_code)
                ran += 1
        return ref, ran
