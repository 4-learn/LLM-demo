"""Offline boundary tests for lesson 16 tool-call arguments."""
import importlib.util
import json
import unittest
from pathlib import Path

MODULE = Path(__file__).with_name("E03_tool_round.py")
spec = importlib.util.spec_from_file_location("E03_tool_round", MODULE)
lesson = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lesson)


class ToolArgumentTests(unittest.TestCase):
    def execute(self, raw):
        return lesson.execute({
            "id": "call_test",
            "function": {"name": "count_events", "arguments": raw},
        })

    def content(self, message):
        return json.loads(message["content"])

    def test_reject_non_object_json_arguments(self):
        for raw in ('[]', 'null', '"text"', '42', 'true'):
            with self.subTest(raw=raw):
                message = self.execute(raw)
                self.assertEqual(message["role"], "tool")
                self.assertEqual(message["tool_call_id"], "call_test")
                self.assertIn("JSON object", self.content(message)["error"])

    def test_reject_missing_required_property(self):
        message = self.execute('{"site":"A"}')
        self.assertIn("缺少必填參數 type", self.content(message)["error"])

    def test_reject_wrong_property_type_enum_and_extra_property(self):
        cases = (
            ('{"site":1,"type":"no_helmet"}', "site 必須是字串"),
            ('{"site":"C","type":"no_helmet"}', "不在允許清單"),
            ('{"site":"A","type":"no_helmet","extra":1}', "未定義的參數 extra"),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertIn(expected, self.content(self.execute(raw))["error"])

    def test_reject_malformed_json_without_raising(self):
        message = self.execute('{"site":')
        self.assertIn("不是合法 JSON", self.content(message)["error"])

    def test_valid_object_executes_and_preserves_call_id(self):
        message = self.execute('{"site":"A","type":"no_helmet"}')
        self.assertEqual(message["tool_call_id"], "call_test")
        self.assertEqual(self.content(message)["count"], 2)

    def test_run_turn_rejects_malformed_or_duplicate_call_ids_without_crashing(self):
        malformed = {"role": "assistant", "tool_calls": [None]}
        messages, status = lesson.run_turn(lambda _messages: malformed, "test")
        self.assertIn("協定錯誤", status)
        self.assertEqual(len(messages), 1)  # no tool executed
        duplicate_ids = {"role": "assistant", "tool_calls": [
            lesson.tool_call("same", "count_events", '{"site":"A","type":"no_helmet"}') ["tool_calls"][0],
            lesson.tool_call("same", "count_events", '{"site":"B","type":"no_helmet"}') ["tool_calls"][0],
        ]}
        messages, status = lesson.run_turn(lambda _messages: duplicate_ids, "test")
        self.assertIn("協定錯誤", status)
        self.assertEqual(len(messages), 1)

    def test_max_rounds_discloses_missing_final_answer(self):
        messages, status = lesson.run_turn(lesson.ScriptedModel([]), "test")
        self.assertIn("尚無最終答覆", status)
        self.assertEqual(messages[-1]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
