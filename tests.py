#!/usr/bin/env python3.11
""""""
from __future__ import annotations
import unittest
import json
import subprocess
import parsing
from lepi import *


class TestParsing(unittest.TestCase):
    pass


with open("tests/expected.json", "r", encoding="utf-8") as file:
    EXPECTED = json.load(file)


def make_func(filename, expected_output):
    def test_filename(self):
        with open(f"tests/{filename}", "r", encoding="utf-8") as file:
            contents = file.read()

        result = parse_to_vm(contents)
        self.assertTrue(result.is_ok)

        vm = result.unwrap()

        max_memdump = 81
        vm.log_execution = lambda *args, **kwargs: None
        vm.run()

        self.assertEqual(vm.buffer, expected_output.pop("buffer"))

        print(vm.registers)

        for register_name, expected_value in expected_output.items():
            self.assertEqual(vm.get_register(getattr(Register, register_name.upper())).as_int(), expected_value)

    return test_filename


for filename, expected_output in EXPECTED.items():
    setattr(TestParsing, f"test_{filename}", make_func(filename, expected_output))

if __name__ == "__main__":
    unittest.main()
