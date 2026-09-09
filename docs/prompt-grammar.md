# Prompt grammar

The default frontend (`aimc/intent/`) is a deterministic semantic parser:
no network, no model, table-driven patterns over a normalised token stream.
Preambles such as "please", "write a program that", "the program should" are
stripped; number words, ordinals and contractions are normalised; `#`/`//`
lines are comments. Sentences are split on `.`, `;`, newlines and
"then"/"and then"; bodies are chained with "and", commas, "then", or by
indented lines after a colon.

* **Print**: `print/output/display/show/say/write/echo E`; quoted literals
  (`\n`, `\t` escapes); bare words become text with casing kept; `print A
  and B` / `A, B` / `A followed by B`; `without a newline`; `separated by a
  comma/space/tab/nothing`; `print a blank line`.
* **Expressions**: infix `+ - * / % ^ ** ( )` and comparisons; English forms
  `the sum/difference/product/quotient of A and B`, `A plus/minus/times/
  divided by/mod/modulo B`, `the remainder when A is divided by B`,
  `A squared/cubed/factorial/doubled/halved/negated/reversed`, `A!`, `A to the
  power of B`, `double/triple/half of A`, `negative A`, `the absolute value/
  square/cube/factorial of A`, `the maximum/largest/minimum/smallest of A, B
  and C`, `the gcd of A and B`, `the number of digits in A`, `the reverse of
  A`, `the Nth fibonacci number`, `the sum of the (even|odd) numbers from A to
  B`, call syntax `max(…) min(…) abs(…) pow(…) gcd(…) fib(…) factorial(…)
  digits(…) reverse(…) prime(…)`, number words, and the pronouns `it` / `that`
  / `the number` / `the result` / `its …` / `their …` resolved from context.
* **Conditions**: `is even/odd/zero/nonzero/positive/negative/prime`, `is
  (not) equal to / greater than / less than / at least / at most / more than
  / fewer than / above / below`, `equals`, `exceeds`, `is divisible by`, `is a
  multiple of`, `is a factor of`, `is between A and B`, combined with
  `and` / `or` / `but` / `not` / `either` / `both` and parentheses.
* **Variables**: `set/initialize x to E`, `let x be E`, `x = E`, `define x as
  E`, `store E in x`, `increase/decrease x by E`, `add E to x`, `subtract E
  from x`, `multiply/divide x by E`, `double/halve/square/negate x`, `x += E`,
  `x++`, `compute E (and print the result)`.
* **Input / exit**: `read a number (into x | called x)`, `read two numbers a
  and b`, `ask the user for a number`, `let x be a number read from input`;
  `exit (with code E)`, `exit with an error`, `return E`.
* **Control flow**: `if C then/,/: S (, otherwise/else S)`, chained `else if`,
  trailing `S if C`, `S unless C`, `while C, S`, `until C, S`, `repeat S until
  C`, `keep doing S while C`, `for each (even|odd) number n from A to B (down
  to | in steps of K | exclusive)`, `for i = A to B`, `for x in A..B`,
  `repeat N times: S`, `S N times`, `S twice`.
* **Idioms**: `print the numbers from A to B`, `the even/odd numbers between A
  and B`, `the squares/cubes of the numbers from A to B`, `the multiples of K
  up to N`, `the first N fibonacci numbers`, `fizzbuzz up to N`, `count (down)
  from A to B`, `print whether X is even or odd / is prime / C`, `the
  multiplication table for N`, `the digits of N`, `the first N primes`, `the
  primes up to N`, `the collatz sequence starting at N`, `read a number and
  print it doubled / its factorial`, `read two numbers and print their sum`.

Not supported (by design, for now): user-defined functions, string values
beyond print literals, floats, arrays. Unrecognised clauses raise
`IntentError` naming the clause; nothing is guessed silently. IR helpers such
as `factorial`, `gcd`, `is_prime`, `fib`, `pow` are emitted as IR functions
(`aimc/intent/stdlib.py`) so they compile through the same backends.
