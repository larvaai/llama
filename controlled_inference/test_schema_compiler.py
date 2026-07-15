import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compile_schema_subset import compile_schema


class SchemaCompilerTests(unittest.TestCase):
    def test_supported_primitives_enum_and_nullable(self):
        schema = {
            "type": "object",
            "properties": {
                "s": {"type": "string", "enum": ["a", "b"]},
                "i": {"type": ["integer", "null"]},
                "b": {"type": "boolean"},
                "n": {"type": "null"},
            },
            "required": ["s", "i", "b", "n"],
            "additionalProperties": False,
        }
        grammar = compile_schema(schema)
        self.assertIn('"true" | "false"', grammar)
        self.assertIn('json-integer | "null"', grammar)
        self.assertIn('"\\\"a\\\"" | "\\\"b\\\""', grammar)

    def test_rejects_additional_properties(self):
        with self.assertRaises(ValueError):
            compile_schema({"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]})

    def test_rejects_optional_property(self):
        with self.assertRaises(ValueError):
            compile_schema({
                "type": "object",
                "properties": {"x": {"type": "string"}, "y": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": False,
            })

    def test_rejects_nested_object(self):
        with self.assertRaises(ValueError):
            compile_schema({
                "type": "object",
                "properties": {"x": {"type": "object"}},
                "required": ["x"],
                "additionalProperties": False,
            })


if __name__ == "__main__":
    unittest.main()
