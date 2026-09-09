"""End-to-end tests for the intent parser: English -> IR -> reference interpreter."""
import unittest

from aimc.intent import IntentError, explain, parse
from aimc.ir import TRAP_EXIT_CODE, TRAP_MESSAGE, Interpreter, digest, from_json, to_json, verify


def run(text, stdin=b""):
    module = parse(text)
    verify(module)
    r = Interpreter(module, stdin=stdin).run()
    return r.stdout.decode(), r.exit_code


class IntentTestCase(unittest.TestCase):
    def check(self, text, stdout, code=0, stdin=b""):
        out, rc = run(text, stdin)
        self.assertEqual(out, stdout, "prompt: %r" % text)
        self.assertEqual(rc, code, "prompt: %r" % text)


class TestPrint(IntentTestCase):
    def test_hello_world_bare(self):
        self.check("print hello world", "hello world\n")

    def test_quoted_literal_keeps_case_and_punctuation(self):
        self.check('Please write a program that prints "Hello, World!"', "Hello, World!\n")

    def test_bare_words_keep_case(self):
        self.check("print Hello World", "Hello World\n")

    def test_print_expression_english(self):
        self.check("print the sum of 3 and 4", "7\n")

    def test_print_infix_precedence(self):
        self.check("print 10 minus 3 times 2. print (10 - 3) * 2", "4\n14\n")

    def test_print_multiple_items(self):
        self.check("print 'x =' followed by 5. print 1 and 2", "x = 5\n1 2\n")

    def test_print_without_newline_and_separator(self):
        self.check("print 1 without a newline. print 2. print 'a' and 5 separated by a comma", "12\na,5\n")

    def test_blank_line(self):
        self.check("print a blank line", "\n")

    def test_print_synonyms(self):
        self.check("output Hello. display the product of 6 and 7. show 5 factorial", "Hello\n42\n120\n")

    def test_escape_in_literal(self):
        self.check('print "a\\tb"', "a\tb\n")


class TestExpressions(IntentTestCase):
    def test_power_and_modulo(self):
        self.check("print 2 to the power of 10, 17 mod 5, 2 ^ 3 ** 2", "1024 2 512\n")

    def test_remainder_phrases(self):
        self.check("print the remainder of 17 divided by 5 and the remainder when 9 is divided by 4", "2 1\n")

    def test_function_call_syntax(self):
        self.check("print max(3, 4), min(3, 4), abs(-2), pow(2, 5), fib(7), factorial(4), gcd(12, 18)",
                   "4 3 2 32 13 24 6\n")

    def test_nth_fibonacci_and_digits(self):
        self.check("print the 10th fibonacci number and the number of digits in -12345 and 1200 reversed", "55 5 21\n")

    def test_largest_of_three(self):
        self.check("print the largest of 3, 9 and 5. print the smallest of 3, 9 and 5", "9\n3\n")

    def test_sum_of_ranges(self):
        self.check("print the sum of the numbers from 1 to 100. print the sum of the odd numbers from 1 to 9. "
                   "print the sum of the first 10 numbers", "5050\n25\n55\n")

    def test_wrapping_arithmetic(self):
        self.check("print 9223372036854775807 plus 1", "-9223372036854775808\n")

    def test_comparison_values(self):
        self.check("set x to 5. print x is greater than 3. print not (3 > 2). print x is prime", "1\n0\n1\n")

    def test_number_words(self):
        self.check("print twenty five plus a hundred", "125\n")


class TestVariables(IntentTestCase):
    def test_assignment_forms(self):
        self.check("set x to 5. let y be x times 2. z = y + 1. define total as 0. store 7 in w. print x, y, z, total, w",
                   "5 10 11 0 7\n")

    def test_updates(self):
        self.check("set x to 1. increment x. x += 2. multiply x by 10. subtract 5 from x. add 3 to x. "
                   "double x. halve x. decrease x by 1. print x", "37\n")

    def test_compute_and_the_result(self):
        self.check("compute 6 times 7 and print the result", "42\n")
        self.check("compute 6 times 7. print the result", "42\n")

    def test_sum_statement(self):
        self.check("sum the numbers from 1 to 100 and print the result", "5050\n")


class TestInput(IntentTestCase):
    def test_read_forms(self):
        self.check("read a number and print its factorial", "120\n", stdin=b"5\n")
        self.check("read x from input. print x", "9\n", stdin=b"9")
        self.check("ask the user for a number and store it in n. print n plus 1", "42\n", stdin=b"41")
        self.check("let n be a number read from input. print n squared", "36\n", stdin=b"6")

    def test_read_two_and_their(self):
        self.check("read two numbers and print their sum", "7\n", stdin=b"3 4")
        self.check("read two numbers a and b. print the maximum of a and b", "9\n", stdin=b"3\n9\n")

    def test_read_three(self):
        self.check("read three numbers and print the maximum of a, b and c", "9\n", stdin=b"3 9 4")

    def test_read_negative_and_eof(self):
        self.check("read a number into x. read a number into y. print x and y", "-12 0\n", stdin=b"  -12")

    def test_read_and_pronoun(self):
        self.check("read a number and print it doubled", "42\n", stdin=b"21")
        self.check("read a number and print whether it is even or odd", "even\n", stdin=b"4")

    def test_third_person_preamble(self):
        self.check("write a program that reads a number n and prints n cubed", "27\n", stdin=b"3")


class TestControlFlow(IntentTestCase):
    def test_if_else_chain(self):
        prog = ("read a number into n. if n is negative, print negative, otherwise if n is zero, "
                "print 'zero', otherwise print positive")
        self.check(prog, "negative\n", stdin=b"-3")
        self.check(prog, "zero\n", stdin=b"0")
        self.check(prog, "positive\n", stdin=b"8")

    def test_trailing_if_and_unless(self):
        self.check("read a number into x. print x if x is even, otherwise print odd", "odd\n", stdin=b"7")
        self.check("set x to 3. unless x is even, print odd", "odd\n")

    def test_else_after_sentence_boundary(self):
        self.check("if 1 is even, print e. otherwise print o.", "o\n")

    def test_compound_condition(self):
        self.check("set x to 7. if x is between 1 and 10 and x is odd then print yes", "yes\n")
        self.check("set x to 7. if x is even or x is greater than 100 then print yes, otherwise print no", "no\n")

    def test_while_loops(self):
        self.check("set x to 1. while x is less than 100, double x. print x", "128\n")
        self.check("set i to 3. until i is zero, print i and decrement i", "3\n2\n1\n")
        self.check("set i to 5. as long as i is positive, print i and subtract 2 from i", "5\n3\n1\n")

    def test_do_until_and_keep(self):
        self.check("set x to 10. repeat print x and decrease x by 3 until x is negative", "10\n7\n4\n1\n")
        self.check("set x to 1. keep doubling x until x exceeds 100. print x", "128\n")
        self.check("set i to 0. do print i and increase i by 1 while i is less than 2", "0\n1\n")

    def test_for_loop_forms(self):
        self.check("for each number n from 1 to 5, print n squared", "1\n4\n9\n16\n25\n")
        self.check("for i = 1 to 3, print i", "1\n2\n3\n")
        self.check("for x in 1..3 print x times 10", "10\n20\n30\n")
        self.check("for each even number from 2 to 6, print it", "2\n4\n6\n")
        self.check("for i from 1 to 10 in steps of 3, print i", "1\n4\n7\n10\n")
        self.check("for i from 10 down to 7, print i", "10\n9\n8\n7\n")
        self.check("loop from 1 to 3 and print it", "1\n2\n3\n")

    def test_repeat_forms(self):
        self.check("repeat 3 times: print hi", "hi\nhi\nhi\n")
        self.check("print hi 3 times", "hi\nhi\nhi\n")
        self.check("print 'hi' twice", "hi\nhi\n")
        self.check("3 times, print go", "go\ngo\ngo\n")

    def test_indented_blocks(self):
        prog = ("for each n from 1 to 15:\n"
                "    if n is divisible by 15, print FizzBuzz\n"
                "    otherwise if n is divisible by 3, print Fizz\n"
                "    otherwise if n is divisible by 5, print Buzz\n"
                "    otherwise print n\n"
                "print done")
        expected = "".join(("FizzBuzz" if n % 15 == 0 else "Fizz" if n % 3 == 0 else "Buzz" if n % 5 == 0 else str(n)) + "\n"
                           for n in range(1, 16)) + "done\n"
        self.check(prog, expected)

    def test_indented_while_and_nested_loops(self):
        self.check("set i to 0\nwhile i is less than 3:\n    print i\n    increment i\nprint done", "0\n1\n2\ndone\n")
        self.check("for i from 1 to 2, for j from 1 to 2, print i times j", "1\n2\n2\n4\n")

    def test_multiline_accumulator(self):
        self.check("set total to 0\nfor i from 1 to 4\n    add i to total\nprint total", "10\n")

    def test_sequence_after_loop(self):
        self.check("print the numbers from 1 to 3 and then print done", "1\n2\n3\ndone\n")


class TestExit(IntentTestCase):
    def test_exit_and_return(self):
        self.check("exit with code 42", "", 42)
        self.check("return 5", "", 5)
        self.check("exit", "", 0)
        self.check("I want a program that prints 42 and then exits with status 1", "42\n", 1)

    def test_conditional_exit(self):
        self.check("read a number into x. exit with code 3 if x is negative. print ok", "", 3, stdin=b"-1")
        self.check("read a number into x. exit with code 3 if x is negative. print ok", "ok\n", 0, stdin=b"1")

    def test_division_by_zero_traps(self):
        module = parse("set y to 0. print 4 divided by y")
        r = Interpreter(module).run()
        self.assertEqual(r.exit_code, TRAP_EXIT_CODE)
        self.assertEqual(r.stderr, TRAP_MESSAGE)
        self.assertEqual(r.stdout, b"")


class TestIdioms(IntentTestCase):
    def test_fizzbuzz(self):
        expected = "".join(("FizzBuzz" if n % 15 == 0 else "Fizz" if n % 3 == 0 else "Buzz" if n % 5 == 0 else str(n)) + "\n"
                           for n in range(1, 16))
        self.check("print fizzbuzz up to 15", expected)
        self.check("play fizzbuzz from 1 to 15", expected)

    def test_ranges(self):
        self.check("print the numbers from 1 to 5", "1\n2\n3\n4\n5\n")
        self.check("print the even numbers between 1 and 10", "2\n4\n6\n8\n10\n")
        self.check("print the squares of the numbers from 1 to 4", "1\n4\n9\n16\n")
        self.check("print the multiples of 7 up to 30", "7\n14\n21\n28\n")
        self.check("count down from 5 to 1", "5\n4\n3\n2\n1\n")
        self.check("count from 1 to 3", "1\n2\n3\n")

    def test_fibonacci(self):
        self.check("print the first 10 fibonacci numbers", "0\n1\n1\n2\n3\n5\n8\n13\n21\n34\n")
        self.check("print the fibonacci sequence up to 20", "0\n1\n1\n2\n3\n5\n8\n13\n")

    def test_primes(self):
        self.check("print the first 5 primes", "2\n3\n5\n7\n11\n")
        self.check("print the primes up to 20", "2\n3\n5\n7\n11\n13\n17\n19\n")
        self.check("print whether 9 is prime. print whether 7 is prime", "not prime\nprime\n")

    def test_whether(self):
        self.check("print whether 7 is even or odd", "odd\n")
        self.check("print whether 3 is greater than 2", "true\n")

    def test_tables_digits_collatz(self):
        self.check("print the multiplication table for 3 up to 4", "3 x 1 = 3\n3 x 2 = 6\n3 x 3 = 9\n3 x 4 = 12\n")
        self.check("print the digits of 1203", "1\n2\n0\n3\n")
        self.check("print the collatz sequence starting at 6", "6\n3\n10\n5\n16\n8\n4\n2\n1\n")
        self.check("reverse the digits of 1200 and print it", "21\n")

    def test_factorial_helper_is_recursive_call(self):
        module = parse("print the factorial of 20")
        names = [f.name for f in module.functions]
        self.assertIn("factorial", names)
        out, rc = run("print the factorial of 20")
        self.assertEqual(out, "2432902008176640000\n")


class TestErrors(unittest.TestCase):
    def test_gibberish(self):
        with self.assertRaises(IntentError):
            parse("frobnicate the widgets")

    def test_unknown_variable(self):
        with self.assertRaises(IntentError) as cm:
            parse("print undefinedvar plus 1")
        self.assertIn("undefinedvar", str(cm.exception))

    def test_unknown_variable_in_condition(self):
        with self.assertRaises(IntentError):
            parse("if x is even print yes")

    def test_empty(self):
        with self.assertRaises(IntentError):
            parse("   ")

    def test_loop_without_body(self):
        with self.assertRaises(IntentError):
            parse("while 1 is even")


class TestSerialization(unittest.TestCase):
    def test_roundtrip_digest(self):
        for text in ("print fizzbuzz up to 15", "read a number and print its factorial",
                     "for i from 1 to 3, print i squared"):
            m = parse(text)
            m2 = from_json(to_json(m))
            self.assertEqual(digest(m), digest(m2))
            self.assertEqual(Interpreter(m, stdin=b"5").run().stdout, Interpreter(m2, stdin=b"5").run().stdout)

    def test_explain(self):
        text = explain("read a number into n. for each i from 1 to n, if i is even, print i, otherwise print odd")
        self.assertIn("read_int", text)
        self.assertIn("for i from 1 to n", text)
        self.assertIn("else:", text)

    def test_deterministic(self):
        self.assertEqual(digest(parse("print the first 5 primes")), digest(parse("print the first 5 primes")))


if __name__ == "__main__":
    unittest.main()
