"""The `python -m aimc` command line, exercised as a subprocess."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

from aimc.codegen import host_target
from aimc.runtime import Sandbox

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def aimc(*args, stdin: bytes = b"") -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "aimc", *args], input=stdin, capture_output=True, cwd=ROOT, timeout=60)


class CliTests(unittest.TestCase):
    def test_ir_text_and_json(self):
        r = aimc("ir", "print 6 times 7", "--text")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(b"mul", r.stdout)
        r = aimc("ir", "print 6 times 7", "--compact")
        doc = json.loads(r.stdout)
        self.assertEqual(doc["aimc_ir"], 1)
        # IR JSON is itself accepted as a source
        r2 = aimc("run", r.stdout.decode(), "--strategy", "interp")
        self.assertEqual((r2.returncode, r2.stdout), (0, b"42\n"))

    def test_run_with_stdin_and_exit_code(self):
        r = aimc("run", "read a number and print it doubled", "--stdin", "21")
        self.assertEqual((r.returncode, r.stdout), (0, b"42\n"), r.stderr)
        r = aimc("run", "exit with code 7")
        self.assertEqual(r.returncode, 7)

    def test_unintelligible(self):
        r = aimc("run", "make me a sandwich")
        self.assertEqual(r.returncode, 3)
        self.assertIn(b"could not understand", r.stderr)

    def test_compile_produces_runnable_executable(self):
        if not Sandbox().strategies(host_target()):
            self.skipTest("host cannot execute its own target")
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "hello")
            r = aimc("compile", "print hello world", "-o", exe)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(b"bytes of code", r.stderr)
            out = subprocess.run([exe], capture_output=True, timeout=10)
            self.assertEqual((out.returncode, out.stdout), (0, b"hello world\n"))
            r = aimc("compile", "print 42", "-f", "raw", "-o", os.path.join(tmp, "code.bin"))
            self.assertEqual(r.returncode, 0)
            self.assertGreater(os.path.getsize(os.path.join(tmp, "code.bin")), 100)

    def test_verify_and_targets(self):
        r = aimc("verify", "read a number and print it doubled", "--stdin", "21", "--expect-stdout", "42")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(b"PASS", r.stdout)
        r = aimc("verify", "print 1", "--expect-stdout", "2")
        self.assertEqual(r.returncode, 1)
        self.assertIn(b"FAIL", r.stdout)
        r = aimc("targets")
        self.assertEqual(r.returncode, 0)
        self.assertIn(b"aarch64-macos", r.stdout)

    def test_explain(self):
        r = aimc("explain", "for each number from 1 to 5, print it squared")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.strip())


if __name__ == "__main__":
    unittest.main()
