"""Lexical normalisation of natural-language program specifications.

Turns free text into a flat token stream that the statement and expression
parsers can pattern-match deterministically:

* curly quotes straightened, comment lines (# or //) dropped
* politeness and preambles ("please write a program that ...") stripped at
  sentence starts; third-person verbs at statement starts ("prints") and
  gerunds ("doubling") mapped to imperatives
* contractions expanded ("isn't" -> "is not")
* string literals kept verbatim (with escapes decoded); everything else
  lower-cased, but the original spelling is kept in `Token.raw` so bare-word
  strings ("print Hello World") preserve casing
* number words merged into numeric tokens ("twenty five" -> 25, "a hundred"
  -> 100); ordinals ("fifth", "5th") become numeric tokens flagged `ordinal`
* "twice"/"once"/"thrice" become "<n> times"
* newlines become `nl` tokens carrying the indentation of the following line
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Token:
    kind: str            # 'word' | 'num' | 'str' | 'punct' | 'nl'
    text: str            # normalised text (lower-case word, digits, literal contents, symbol)
    raw: str             # original spelling
    value: int = 0       # numeric value for 'num'
    ordinal: bool = False
    indent: int = 0      # indentation of the line this token sits on
    line: int = 1

    def __repr__(self) -> str:
        return "%s(%r)" % (self.kind, self.text)


_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
          "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000}
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
             "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
             "fifteenth": 15, "twentieth": 20, "hundredth": 100}
_MULTIPLES = {"once": 1, "twice": 2, "thrice": 3}

_CONTRACTIONS = {
    "isn't": "is not", "aren't": "are not", "doesn't": "does not", "don't": "do not",
    "didn't": "did not", "can't": "cannot", "won't": "will not", "wasn't": "was not",
    "hasn't": "has not", "haven't": "have not", "shouldn't": "should not", "it's": "it is",
    "that's": "that is", "let's": "let us", "i'd": "i would", "you'd": "you would",
    "there's": "there is", "what's": "what is", "wouldn't": "would not", "couldn't": "could not",
}

# third-person verb -> imperative, applied at statement starts
_THIRD_PERSON = {
    "prints": "print", "outputs": "output", "displays": "display", "shows": "show", "reads": "read",
    "computes": "compute", "calculates": "calculate", "sets": "set", "exits": "exit",
    "returns": "return", "counts": "count", "sums": "sum", "asks": "ask", "gets": "get",
    "takes": "take", "loops": "loop", "repeats": "repeat", "adds": "add", "subtracts": "subtract",
    "multiplies": "multiply", "divides": "divide", "doubles": "double", "halves": "halve",
    "increments": "increment", "decrements": "decrement", "increases": "increase",
    "decreases": "decrease", "writes": "write", "says": "say", "echoes": "echo", "lists": "list",
    "stores": "store", "assigns": "assign", "initializes": "initialize", "initialises": "initialise",
    "defines": "define", "declares": "declare", "keeps": "keep", "plays": "play", "reverses": "reverse",
    "squares": "square", "negates": "negate", "evaluates": "evaluate", "finds": "find",
}
_GERUNDS = {
    "printing": "print", "outputting": "output", "displaying": "display", "showing": "show",
    "doubling": "double", "halving": "halve", "adding": "add", "subtracting": "subtract",
    "incrementing": "increment", "decrementing": "decrement", "increasing": "increase",
    "decreasing": "decrease", "multiplying": "multiply", "dividing": "divide", "reading": "read",
    "setting": "set", "squaring": "square", "negating": "negate", "computing": "compute",
    "repeating": "repeat", "exiting": "exit", "returning": "return", "writing": "write",
    "saying": "say", "echoing": "echo", "asking": "ask", "getting": "get", "tripling": "triple",
}

_VERB_RE = "|".join(sorted(_THIRD_PERSON) + ["then", "should", "must", "will", "needs to", "has to",
                                             "print", "read", "set", "output", "display", "show",
                                             "exit", "return", "compute", "count", "loop", "repeat",
                                             "ask", "get", "for", "if", "while", "let", "increase",
                                             "decrease", "add", "subtract", "double", "halve"])

_PREAMBLES = [
    r"(?:please|kindly)[,\s]+",
    r"(?:can|could|would|will) you(?: please)?\s+",
    r"i(?:'d| would)? (?:want|need|like|would like)(?: you)?(?: to)?\s+",
    r"(?:write|create|make|build|generate|implement|code|develop|produce|design|give me|show me|compile)"
    r"(?: me)?(?: up)?(?: a| an| the)?(?: simple| small| short| tiny| little| quick| basic)?"
    r"(?: (?:python|c|rust|command line|cli|console|terminal))? ?(?:program|script|code|snippet|application|app|function|routine)"
    r"(?: that| which| to| so that it| such that it)?\s*",
    r"(?:a |an |the )?(?:program|script|code) (?:that|which|should|to|will|must|needs to|has to)\s+",
    r"(?:the |this )?(?:program|code|script|it) (?:should|must|needs to|has to|will|then)\s+",
    r"(?:the |this )?(?:program|code|script|it) (?=(?:" + _VERB_RE + r")\b)",
    r"(?:first|second|third|next|then|now|also|lastly|finally|after that|afterwards|to start|to begin|to finish)(?: of all)?,\s*",
    r"(?:here is|here's) what (?:i want|it should do|to do):?\s*",
    r"(?:i want|i need|i would like)(?: it)?(?: to)?\s+",
    r"(?:make sure|ensure)(?: that)?(?: it| the program)?\s+",
]
_PREAMBLE_RES = [re.compile(r"(?m)^(?P<lead>[ \t]*)(?:" + p + ")", re.I) for p in _PREAMBLES]
_PREAMBLE_RES_MID = [re.compile(r"(?<=[.;])(?P<lead>[ \t]*)(?:" + p + ")", re.I) for p in _PREAMBLES]

_TOKEN_RE = re.compile(r'''
    (?P<ws>[ \t]+)
  | (?P<dstr>"(?:[^"\\]|\\.)*")
  | (?P<sstr>(?<![A-Za-z0-9])'(?:[^'\\]|\\.)*'(?![A-Za-z0-9]))
  | (?P<num>\d+(?:st|nd|rd|th)?)(?![A-Za-z])
  | (?P<word>[A-Za-z_][A-Za-z0-9_]*(?:'[A-Za-z]+)?)
  | (?P<punct>\*\*|==|!=|<=|>=|\+=|-=|\*=|/=|%=|\+\+|--|\.\.|:=|[-+*/%^()=<>,.;:!?\[\]{}&|~])
  | (?P<other>.)
''', re.X)

_ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", '"': '"', "'": "'", "r": "\r", "0": "\0"}


def _unescape(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            out.append(_ESCAPES.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def strip_preambles(text: str) -> str:
    for _ in range(8):
        before = text
        for rx in _PREAMBLE_RES + _PREAMBLE_RES_MID:
            text = rx.sub(lambda m: m.group("lead"), text)
        if text == before:
            break
    return text


def normalize_text(text: str) -> str:
    text = (text.replace("“", '"').replace("”", '"').replace("‘", "'")
            .replace("’", "'").replace("—", "-").replace("–", "-"))
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    lines = [ln for ln in text.split("\n") if not re.match(r"^\s*(#|//)", ln)]
    text = "\n".join(lines)
    text = re.sub(r"\bnon-zero\b", "nonzero", text, flags=re.I)
    text = re.sub(r"\bfizz-buzz\b", "fizzbuzz", text, flags=re.I)
    text = re.sub(r"\bstd(?:in|out)\b", lambda m: m.group(0).lower(), text)
    return strip_preambles(text)


def _split_word(raw: str) -> List[str]:
    low = raw.lower()
    if "'" in low:
        if low in _CONTRACTIONS:
            return _CONTRACTIONS[low].split()
        if low.endswith("'s"):
            return [low[:-2]]
        return [low.replace("'", "")]
    return [low]


def _scan_line(line: str, lineno: int, indent: int) -> List[Token]:
    toks: List[Token] = []
    prev_ws = True
    for m in _TOKEN_RE.finditer(line):
        kind = m.lastgroup
        s = m.group(0)
        if kind == "ws":
            prev_ws = True
            continue
        if kind in ("dstr", "sstr"):
            toks.append(Token("str", _unescape(s[1:-1]), s, indent=indent, line=lineno))
        elif kind == "num":
            mm = re.match(r"(\d+)(st|nd|rd|th)?$", s)
            toks.append(Token("num", mm.group(1), s, value=int(mm.group(1)),
                              ordinal=bool(mm.group(2)), indent=indent, line=lineno))
        elif kind == "word":
            for w in _split_word(s):
                toks.append(Token("word", w, s if len(_split_word(s)) == 1 else w, indent=indent, line=lineno))
        elif kind == "punct":
            if s == "!":
                prev = toks[-1] if toks else None
                if not prev_ws and prev is not None and (prev.kind == "num" or prev.text == ")"):
                    toks.append(Token("punct", "!", s, indent=indent, line=lineno))
                else:
                    toks.append(Token("punct", ".", s, indent=indent, line=lineno))
            elif s == "?":
                toks.append(Token("punct", ".", s, indent=indent, line=lineno))
            else:
                toks.append(Token("punct", s, s, indent=indent, line=lineno))
        else:  # stray character
            if s.strip():
                toks.append(Token("punct", s, s, indent=indent, line=lineno))
        prev_ws = False
    return toks


def _merge_numbers(toks: List[Token]) -> List[Token]:
    out: List[Token] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        starts_number = t.kind == "word" and (
            t.text in _UNITS or t.text in _TENS or t.text in _SCALES
            or (t.text in ("a", "an") and i + 1 < len(toks) and toks[i + 1].kind == "word"
                and toks[i + 1].text in _SCALES))
        if not starts_number:
            out.append(t)
            i += 1
            continue
        j = i
        total = 0
        current = 0
        raws: List[str] = []
        seen_unit = False
        while j < len(toks) and toks[j].kind == "word":
            w = toks[j].text
            if j == i and w in ("a", "an"):
                current = 1
            elif w in _UNITS:
                if seen_unit:
                    break
                current += _UNITS[w]
                seen_unit = True
            elif w in _TENS:
                if seen_unit:
                    break
                current += _TENS[w]
            elif w in _SCALES:
                current = (current or 1) * _SCALES[w]
                if _SCALES[w] >= 1000:
                    total += current
                    current = 0
                seen_unit = False
            else:
                break
            raws.append(toks[j].raw)
            j += 1
        value = total + current
        out.append(Token("num", str(value), " ".join(raws), value=value, indent=t.indent, line=t.line))
        i = j
    return out


_STMT_BOUNDARY_WORDS = {"and", "then", "also", "next", "finally", "otherwise", "else", "afterwards"}


def _postprocess(toks: List[Token]) -> List[Token]:
    out: List[Token] = []
    for idx, t in enumerate(toks):
        if t.kind == "word":
            if t.text in _ORDINALS:
                n = _ORDINALS[t.text]
                out.append(Token("num", str(n), t.raw, value=n, ordinal=True, indent=t.indent, line=t.line))
                continue
            if t.text in _MULTIPLES:
                n = _MULTIPLES[t.text]
                out.append(Token("num", str(n), t.raw, value=n, indent=t.indent, line=t.line))
                out.append(Token("word", "times", "times", indent=t.indent, line=t.line))
                continue
            if t.text in _GERUNDS:
                out.append(Token("word", _GERUNDS[t.text], t.raw, indent=t.indent, line=t.line))
                continue
            if t.text in _THIRD_PERSON:
                prev = out[-1] if out else None
                at_start = (prev is None or prev.kind == "nl"
                            or (prev.kind == "punct" and prev.text in (".", ";", ",", ":"))
                            or (prev.kind == "word" and prev.text in _STMT_BOUNDARY_WORDS))
                if at_start:
                    out.append(Token("word", _THIRD_PERSON[t.text], t.raw, indent=t.indent, line=t.line))
                    continue
        out.append(t)
    return out


def tokenize(text: str) -> List[Token]:
    text = normalize_text(text)
    toks: List[Token] = []
    for lineno, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        line_toks = _scan_line(line.strip(), lineno, indent)
        if toks and line_toks:
            toks.append(Token("nl", "\n", "\n", indent=indent, line=lineno))
        toks.extend(line_toks)
    toks = _merge_numbers(toks)
    toks = _postprocess(toks)
    return toks


def text_of(t: Token) -> str:
    """Canonical text used for regex matching of statement shapes."""
    if t.kind == "num":
        if t.ordinal:
            return "first" if t.value == 1 else t.text + "th"
        return t.text
    if t.kind == "str":
        return "\x01"
    return t.text
