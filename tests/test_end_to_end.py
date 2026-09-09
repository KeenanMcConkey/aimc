"""Acceptance suite: English prompt -> IR -> raw machine code -> native execution -> asserted I/O.

Every corpus entry goes through `aimc.verify.verify_prompt`, which parses the
prompt, verifies and round-trips the IR, runs the interpreter, compiles for
every supported target, executes the machine code on every target this host
can run (real executable *and* JIT loader), and demands byte-identical
stdout / stderr / exit status across all of them and against the
expectation given here.
"""
from __future__ import annotations

import unittest

from aimc.ir import TRAP_EXIT_CODE, TRAP_MESSAGE
from aimc.verify import runnable_targets, verify_prompt
from tests.support import SANDBOX


def lines(*xs) -> bytes:
    return "".join(f"{x}\n" for x in xs).encode()


# (prompt, stdin, expected stdout, expected exit code)
CORPUS = [
    ("print hello world", b"", b"hello world\n", 0),
    ('Print "Hello, World!"', b"", b"Hello, World!\n", 0),
    ("print 6 times 7", b"", b"42\n", 0),
    ("print (3 + 4) * 2", b"", b"14\n", 0),
    ("print 2 to the power of 10", b"", b"1024\n", 0),
    ("print the factorial of 20", b"", b"2432902008176640000\n", 0),
    ("print the gcd of 48 and 18", b"", b"6\n", 0),
    ("print the largest of 3, 9 and 5", b"", b"9\n", 0),
    ("print the number of digits in 12345", b"", b"5\n", 0),
    ("print the reverse of 1230", b"", b"321\n", 0),
    ("print 9223372036854775807 plus 1", b"", b"-9223372036854775808\n", 0),
    ("print the numbers from 1 to 5", b"", lines(1, 2, 3, 4, 5), 0),
    ("print the even numbers between 1 and 10", b"", lines(2, 4, 6, 8, 10), 0),
    ("print the first 10 fibonacci numbers", b"", lines(0, 1, 1, 2, 3, 5, 8, 13, 21, 34), 0),
    ("print fizzbuzz up to 15", b"", lines(1, 2, "Fizz", 4, "Buzz", "Fizz", 7, 8, "Fizz", "Buzz", 11, "Fizz", 13, 14, "FizzBuzz"), 0),
    ("print the primes up to 20", b"", lines(2, 3, 5, 7, 11, 13, 17, 19), 0),
    ("print the collatz sequence starting at 6", b"", lines(6, 3, 10, 5, 16, 8, 4, 2, 1), 0),
    ("count down from 3 to 1", b"", lines(3, 2, 1), 0),
    ("for each number i from 1 to 5, print i squared", b"", lines(1, 4, 9, 16, 25), 0),
    ("repeat 3 times: print hi", b"", lines("hi", "hi", "hi"), 0),
    ("set x to 10. while x is greater than 0, print x then subtract 3 from x.", b"", lines(10, 7, 4, 1), 0),
    ("Set total to 0. For each number i from 1 to 100, if i is divisible by 3 or i is divisible by 5, add i to total. Print total.",
     b"", b"2418\n", 0),
    ("set n to 5\nset acc to 1\nwhile n is greater than 1:\n    multiply acc by n\n    decrease n by 1\nprint acc\n", b"", b"120\n", 0),
    ("read a number and print it doubled", b"21\n", b"42\n", 0),
    ("read two numbers and print their product", b"6 7\n", b"42\n", 0),
    ("read a number n. print whether n is even or odd", b"7\n", b"odd\n", 0),
    ("read a number. if it is negative, exit with code 1. otherwise print it.", b"-5\n", b"", 1),
    ("read a number. if it is negative, exit with code 1. otherwise print it.", b"5\n", b"5\n", 0),
    ("Read a number n from input. Print the factorial of n.", b"10\n", b"3628800\n", 0),
    ('Read a number from input. If it is prime, print "prime" and exit with code 0, otherwise print "composite" and exit with code 1.',
     b"97\n", b"prime\n", 0),
    ('Read a number from input. If it is prime, print "prime" and exit with code 0, otherwise print "composite" and exit with code 1.',
     b"91\n", b"composite\n", 1),
    ("exit with code 42", b"", b"", 42),
    ("print 7 divided by 0", b"", b"", TRAP_EXIT_CODE),
]


class EndToEndTests(unittest.TestCase):
    def test_host_runs_at_least_one_target(self):
        self.assertTrue(runnable_targets(SANDBOX), "no executable target on this host")

    def test_corpus(self):
        for prompt, stdin, out, code in CORPUS:
            with self.subTest(prompt=prompt, stdin=stdin):
                report = verify_prompt(prompt, stdin=stdin, expected_stdout=out, expected_exit=code,
                                       strategies=("exe", "jit"), sandbox=SANDBOX)
                self.assertTrue(report.passed, report.summary())
                native = [k for k in report.runs]
                self.assertTrue(native, "no native execution happened")
                for key, res in report.runs.items():
                    self.assertEqual(res.stdout, out, key)
                    self.assertEqual(res.exit_code, code, key)

    def test_trap_reports_on_stderr_natively(self):
        report = verify_prompt("print 7 divided by 0", strategies=("exe",), sandbox=SANDBOX)
        self.assertTrue(report.passed, report.summary())
        for res in report.runs.values():
            self.assertEqual(res.stderr, TRAP_MESSAGE)
            self.assertEqual(res.exit_code, TRAP_EXIT_CODE)

    def test_unintelligible_prompt_fails_at_intent_stage(self):
        report = verify_prompt("make me a sandwich with extra pickles", sandbox=SANDBOX)
        self.assertFalse(report.passed)
        self.assertEqual(report.checks[0].name, "intent")
        self.assertFalse(report.checks[0].passed)
        self.assertEqual(len(report.checks), 1)  # nothing downstream ran

    def test_ir_is_deterministic(self):
        from aimc.intent import parse
        from aimc.ir import digest
        a = digest(parse("print the first 10 fibonacci numbers"))
        b = digest(parse("print the first 10 fibonacci numbers"))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
