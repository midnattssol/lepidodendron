#!/usr/bin/env python3.11
""""""
import argparse
import functools as ft
import itertools as it
import pathlib as p
import typing as t

import more_itertools as mit
import regex as re
import terscii
from vm import Mode, Opcode, Register, Trits, Tryte, VirtualMachine, pad_to_tryte, hept_dump

from agave import *

MACROS = {
    # ("incr"): (["$0"], ["add $0,1"]),
    # ("decr"): (["$0"], ["add $0,-1"]),
    # ("sub"): (["$0", "$1"], ["neg $1", "add $0,$1", "neg $1"]),
}


def translate(mapping: dict[str, str], data: str) -> str:
    """Apply multiple replacements simultaneously."""
    items = re.finditer("|".join(f"({re.escape(i)})" for i in mapping), data)
    items = reversed(list(items))

    for occurrence in items:
        data = data[: occurrence.start()] + mapping[occurrence.group()] + data[occurrence.end() :]

    return data


def parse_integer(input_str: str) -> t.Optional[int]:
    # Allow arbitrary underscores for readability.
    input_str = input_str.replace("_", "")
    sign = -1 if input_str.startswith("-") else 1
    input_str = input_str.removeprefix("-")

    number_literals = {
        r"^(\d+)$": 10,
        r"^0b([01]+)$": 2,
        r"^0x([0-9a-fA-F]+)$": 16,
    }

    for k, v in number_literals.items():
        if match := re.match(k, input_str):
            return Just(sign * int(match.group(1), v))

    # Handle balanced number systems on its own.
    if match := re.match(r"^0t([01T]+)$", input_str):
        return Just(sign * Tryte.from_balanced_ternary(match.group(1)).as_int())

    # Handle heptavintimal.
    if match := re.match(r"^0h([0-9a-zA-Z]+)$", input_str):
        return Just(sign * Tryte.from_balanced_heptavintimal(match.group(1)).as_int())

    return Nil


def parse_register(rhs):
    return getattr(Register, rhs.upper())


def parse_rhs(rhs, identifiers):
    if rhs in identifiers:
        return parse_rhs(str(identifiers[rhs]), identifiers)

    if (num := parse_integer(rhs)).is_some:
        num = num.unwrap()

        # Store small values in the immediate representation.
        if fits_into_n_trits(num, 2):
            rhs = (Mode.IMMEDIATE, Trits.trits_from_int(num, 2)), Nil
            return rhs

        # If it's a big value, the next tryte is allocated to it.
        rhs = (Mode.NEXT_TRYTE, Trits.trits_from_int(0, 2)), Just(num)
        return rhs

    if not rhs.startswith("*"):
        # If the item isn't some kind of pointer, it is a register.
        rhs = (Mode.REGISTER, parse_register(rhs)), Nil
        return rhs

    # The value is a pointer offset.
    if "->" in rhs:
        rhs, num = rhs.split("->")
        num = parse_integer(num).unwrap()
        rhs = (Mode.PTR_OFFSET, parse_register(rhs)), Just(num)
        return rhs

    # The value is an indirect value.
    # Check if the value is some kind of incrementing/decrementing operation,
    # or otherwise pass it as a simple pointer.
    rhs = rhs.removeprefix("*")
    regexes = {
        r"^(.*)--$": Mode.PTR_POSTDECREMENT,
        r"^(.*)\+\+$": Mode.PTR_POSTINCREMENT,
        r"^\+\+(.*)$": Mode.PTR_PREINCREMENT,
        r"^--(.*)$": Mode.PTR_PREDECREMENT,
    }

    for regex, mode in regexes.items():
        if not (match := re.match(regex, rhs)):
            continue

        register = match.group(1)
        rhs = (mode, parse_register(register)), Nil
        return rhs

    rhs = (Mode.PTR, parse_register(rhs)), Nil
    return rhs


def get_arity(opcode: Opcode) -> int:
    """Get the arity of an operation."""
    if opcode == Opcode.NOOP:
        return 0
    if opcode == Opcode.OUTPUT:
        return 1
    return 2


def parse_one_macro(stream: list, identifiers: dict[str, int], instruction: str) -> None:
    expected, data = MACROS[instruction]
    assert len(args) == len(expected)
    mapping = dict(zip(expected, args))

    for instr in data:
        new_instr = translate(mapping, instr)
        parse_one(stream, identifiers, new_instr)

    return


def parse_one(stream: list, identifiers: dict[str, int], instruction: str) -> None:
    """Parse an instruction and add data into the stream."""
    if not instruction:
        return

    # Set the value of labels.
    if instruction.startswith("@"):
        identifiers[instruction] = len(stream)
        return

    # instruction, args =
    attempted_split = instruction.split(maxsplit=1)

    instruction = attempted_split[0]
    args = attempted_split[1].split(",") if len(attempted_split) > 1 else []
    args = list(map(str.strip, args))

    # Look up and expand macros.
    if instruction in MACROS:
        parse_one_macro(stream, identifiers, instruction)
        return

    opcode = getattr(Opcode, instruction.upper())
    assert args

    if get_arity(opcode) != len(args):
        raise ValueError(f"Wrong number of arguments for opcode {opcode} with arity {get_arity(opcode)}")

    is_unary = len(args) == 1

    # Try to parse the last argument.
    (mode, data), num = parse_rhs(args[-1], identifiers)
    first_argument = Trits.zero(2) if is_unary else parse_register(args[0])

    result = ft.reduce(Trits.extend, [opcode, first_argument, mode, data], Trits.empty())
    assert len(result.trits) == 9
    result = Tryte(result.trits)
    stream.append(result)

    modes_allocating_next_tryte = [Mode.NEXT_TRYTE, Mode.PTR_OFFSET]
    next_tryte_is_allocated = mode in modes_allocating_next_tryte

    if next_tryte_is_allocated:
        stream.append(Tryte.from_int(num.unwrap()))


def separate_sections(raw_data: str) -> dict[str, str]:
    # Extract the section name and the section data as the two groups.
    # From this, the sections are stored as a dictionary.
    section_pattern = r"^\.(\w+)\s*([\s\S]+?)(?=^\.|\Z)"
    sections = re.finditer(section_pattern, raw_data, re.M)
    sections = dict(map(re.Match.groups, sections))

    if "data" not in sections:
        sections["data"] = ""

    return sections


def assemble_program(raw_data) -> (list[Tryte], list[Tryte]):
    """Assemble a program."""
    sections = separate_sections(raw_data)
    assert "data" in sections and "code" in sections

    data, identifiers = parse_data_section(sections["data"])
    code_text = sections["code"]

    instructions = code_text.strip().split("\n")
    instructions = map(str.strip, instructions)
    # Remove comments
    instructions = (i.split(";", maxsplit=1)[0] for i in instructions)
    instructions = list(instructions)

    # Store some default value for the labels, and then run the whole
    # stream once just to store the labels. This is done to find out
    # what the value of the label should be, since the expansion width
    # of instructions is not fixed.
    labels = re.findall(r"(@.*)\n", code_text)
    identifiers |= dict(zip(labels, it.repeat(-999)))

    stream = []
    for opcode in instructions:
        parse_one(stream, identifiers, opcode)

    stream = []
    for opcode in instructions:
        parse_one(stream, identifiers, opcode)

    return data, stream


def parse_data_section(data: str) -> Result[dict[str, t.Any]]:
    """Parse the data section."""
    lines = data.splitlines()
    lines = map(str.strip, lines)
    lines = filter(None, lines)

    data = []
    pointers = {}

    for line in lines:
        name, result = map(str.strip, line.split("="))
        pointers[name] = len(data)

        assert result.endswith('"') == result.startswith('"')

        # The constant is a string.
        # Add null to the end of it to terminate it.
        if result.startswith('"'):
            result = result.removeprefix('"').removesuffix('"')
            result = result.replace("\\0", "\0")
            data.extend(Tryte.from_int(terscii.to_terscii(i)) for i in result)
            continue

        # The constant is a number.
        if (num := parse_integer(result)).is_some:
            data.append(Tryte.from_int(num.unwrap()))
            continue

        # The constant is a list of numbers.
        if result.startswith("[") and result.endswith("]"):
            items = result.removeprefix("[").removesuffix("]").split()
            items = map(parse_integer, items)
            items = Maybe.lift_iter(items).unwrap()

            data.extend(map(Tryte.from_int, items))
            continue

        raise Err(f"Unparsable line {line} in source file.")

    return Ok(data, pointers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a TASM file.")
    parser.add_argument("file", metavar="F", type=p.Path, help="the file to read data from")

    args = parser.parse_args()

    with args.file.open("r", encoding="utf-8") as file:
        contents = file.read()

    data, program = assemble_program(contents)

    vm = VirtualMachine(program)
    vm.memory[: len(data)] = data

    # Set the stack pointer
    vm.set_register(Register.SP, Tryte.from_int(len(data)))

    max_memdump = 81

    vm.run()
    print()
    print("===| Program |===\n")
    print(hept_dump(vm.program))
    print("===| Registers |===\n")
    print(hept_dump(vm.registers))
    print("===| Memory |===\n")
    print(hept_dump(vm.memory[:max_memdump]))
    print("===| Output buffer |===\n")
    print(vm.buffer)


if __name__ == "__main__":
    main()
