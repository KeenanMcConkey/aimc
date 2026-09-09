"""The model-backed frontend, exercised with a stub client (no network)."""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from aimc.intent.llm import IR_SPEC, ModelFrontendError, extract_json, parse_with_model
from aimc.ir import Interpreter

GOOD = {
    "aimc_ir": 1, "entry": 0, "strings": ["hi\n"],
    "functions": [{"name": "main", "params": 0, "slots": 3, "blocks": [[
        [6, 1, None, [], 0, []],
        [1, 0, 0, [], 6, []], [1, 0, 1, [], 7, []], [3, 2, 2, [0, 1], 0, []],
        [6, 0, None, [2], 0, []], [1, 0, 0, [], 10, []], [6, 2, None, [0], 0, []],
        [18, 0, None, [], 0, []],
    ]]}],
}
BAD = json.loads(json.dumps(GOOD))
BAD["functions"][0]["blocks"][0][3] = [3, 2, 2, [0, 5], 0, []]   # slot 5 out of range


class StubClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        stop, text = self.replies.pop(0)
        return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)])


class ModelFrontendTests(unittest.TestCase):
    def test_extract_json_variants(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), '{"a": 1}')
        self.assertEqual(extract_json('Here you go: {"a": {"b": 2}} done'), '{"a": {"b": 2}}')
        with self.assertRaises(ModelFrontendError):
            extract_json("no json here")

    def test_good_reply_runs(self):
        client = StubClient([("end_turn", "```json\n" + json.dumps(GOOD) + "\n```")])
        m = parse_with_model("print hi then 42", client=client)
        self.assertEqual(Interpreter(m).run().stdout, b"hi\n42\n")
        call = client.calls[0]
        self.assertEqual(call["model"], "claude-opus-5")
        self.assertEqual(call["system"], IR_SPEC)
        self.assertEqual(call["fallbacks"], "default")
        self.assertEqual(call["messages"][0], {"role": "user", "content": "print hi then 42"})

    def test_repair_round_feeds_back_verifier_error(self):
        client = StubClient([("end_turn", json.dumps(BAD)), ("end_turn", json.dumps(GOOD))])
        m = parse_with_model("x", client=client, max_repairs=1)
        self.assertEqual(Interpreter(m).run().stdout, b"hi\n42\n")
        self.assertEqual(len(client.calls), 2)
        followup = client.calls[1]["messages"]
        self.assertEqual(followup[1]["role"], "assistant")
        self.assertIn("out of range", followup[2]["content"])

    def test_gives_up_after_repairs(self):
        client = StubClient([("end_turn", json.dumps(BAD))] * 3)
        with self.assertRaisesRegex(ModelFrontendError, "unverifiable IR after 3 attempts"):
            parse_with_model("x", client=client, max_repairs=2)

    def test_refusal(self):
        client = StubClient([("refusal", "")])
        with self.assertRaisesRegex(ModelFrontendError, "declined"):
            parse_with_model("x", client=client)


if __name__ == "__main__":
    unittest.main()
