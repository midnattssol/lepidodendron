#!/usr/bin/env python3.11
""""""
import argparse
import functools as ft
import pathlib as p
import typing as t

import more_itertools as mit
import regex as re
import terscii
from vm import Mode, Opcode, Register, Trits, Tryte, VirtualMachine, output_info

MACROS = {
    # ("incr"): (["$0"], ["add $0,1"]),
    # ("decr"): (["$0"], ["add $0,-1"]),
    # ("sub"): (["$0", "$1"], ["neg $1", "add $0,$1", "neg $1"]),
}


def hept_dump(mem) -> str:
    """Dump a heptavintimal representation of the entire program memory hexdump-style."""
    chunksize = 9
    chunks = mit.chunked(mem, chunksize)
    out = ""

    ptr = Tryte.from_int(0)
    sep = "   "

    fn = Tryte.dump_heptavintimal
    default_repr_subchunk = "–" * len(fn(Tryte.from_int(0)))

    def h(subchunk):
        return (fn(i) if i is not None else default_repr_subchunk for i in subchunk)

    for chunk in chunks:
        chunk = list(chunk)
        chunk = mit.padded(chunk, None, 9)

        subchunks = mit.chunked(chunk, 3)

        chunk_str = ""
        chunk_as_terscii = ""

        for subchunk in subchunks:
            chunk_heptavintimal = " ".join(h(subchunk))
            chunk_str += chunk_heptavintimal + sep

            chunk_as_terscii += "".join(
                "–"
                if i is None
                else terscii.from_terscii(i.as_int())
                if terscii.is_printable(i.as_int())
                and (terscii.from_terscii(i.as_int()) == " " or not terscii.is_whitespace(i.as_int()))
                else "·"
                for i in subchunk
            )

        out += Tryte.from_int(0).dump_heptavintimal() + ptr.dump_heptavintimal() + sep
        out += chunk_str.strip() + sep
        out += "|" + chunk_as_terscii + "|"
        out += "\n"

        ptr = Tryte.from_int(ptr.as_int() + chunksize)

    return out.strip() + "\n"


def is_representable(num: int, k: int) -> bool:
    """Find out if a number is representable in k-trit balanced ternary."""
    lower = Trits.from_ternary("T" * k)
    upper = Trits.from_ternary("1" * k)
    return num <= upper.as_int() and num >= lower.as_int()


def translate(mapping: dict[str, str], data: str) -> str:
    """Apply multiple replacements simultaneously."""
    items = re.finditer("|".join(f"({re.escape(i)})" for i in mapping), data)
    items = reversed(list(items))

    for occurrence in items:
        data = data[: occurrence.start()] + mapping[occurrence.group()] + data[occurrence.end() :]

    return data


def parse_integer(input_str: str) -> t.Optional[int]:
    # Allow arbitrary underscores for readability
    input_str = input_str.replace("_", "")
    sign = 1

    if input_str.startswith("-"):
        sign = -1
        input_str = input_str.removeprefix("-")

    number_literals = {
        r"^([\d_]+)$": 10,
        r"^0b([_01]+)$": 2,
        r"^0x([_0-9a-fA-F]+)$": 16,
    }

    for k, v in number_literals.items():
        if match := re.match(k, input_str):
            return sign * int(match.group(1), v)

    if match := re.match(r"^0t([01T]+)$", input_str):
        return sign * Tryte.from_ternary(match.group(1)).as_int()

    return None


def parse_register(rhs):
    return getattr(Register, rhs.upper())


def parse_rhs(rhs, identifiers):
    if rhs in identifiers:
        return parse_rhs(str(identifiers[rhs]), identifiers)

    if (num := parse_integer(rhs)) is not None:
        # if is_representable(num, 2):
        #     rhs = (Mode.IMMEDIATE, Trits.trits_from_int(num, 2)), num
        #     return rhs

        rhs = (Mode.NEXT_TRYTE, Trits.trits_from_int(0, 2)), num
        return rhs

    if not rhs.startswith("*"):
        # The item is a register
        rhs = (Mode.REGISTER, getattr(Register, rhs.upper())), None
        return rhs

    # The value is a pointer offset.
    if "->" in rhs:
        rhs, num = rhs.split("->")
        rhs = (Mode.PTR_OFFSET, getattr(Register, rhs.upper())), None
        return rhs

    # The value is an indirect value
    rhs = rhs.removeprefix("*")
    regexes = {
        r"^(.*)--$": Mode.PTR_POSTDECREMENT,
        r"^(.*)\+\+$": Mode.PTR_POSTINCREMENT,
        r"^\+\+(.*)$": Mode.PTR_PREINCREMENT,
        r"^--(.*)$": Mode.PTR_PREDECREMENT,
    }

    for k, v in regexes.items():
        if not (match := re.match(k, rhs)):
            continue

        register = match.group(1)
        rhs = (v, getattr(Register, register.upper())), None
        return rhs

    rhs = (Mode.PTR, getattr(Register, rhs.upper())), None
    return rhs


def get_arity(opcode):
    if opcode == Opcode.OUTPUT:
        return 1
    return 2


def parse_one(stream: list, identifiers: dict[str, int], instruction: str) -> None:
    """Parse an instruction and add data into the stream."""
    if not instruction:
        return

    if instruction.startswith("@"):
        identifiers[instruction] = len(stream)
        return

    instruction, args_str = instruction.split(maxsplit=1)

    args_str = args_str.split(",")
    args = list(map(str.strip, args_str))

    # Look up and expand macros.
    if instruction in MACROS:
        expected, data = MACROS[instruction]
        assert len(args) == len(expected)
        mapping = dict(zip(expected, args))

        for instr in data:
            new_instr = translate(mapping, instr)
            parse_one(stream, identifiers, new_instr)

        return

    opcode = getattr(Opcode, instruction.upper())
    assert args

    if get_arity(opcode) != len(args):
        raise ValueError("Wrong number of args")

    if len(args) == 1:
        arbitrary_2_trit_width_data = Trits.zero(2)
        args.insert(0, arbitrary_2_trit_width_data)

    elif len(args) == 2:
        args[0] = parse_register(args[0])

    # Try to parse the 2nd argument.
    assert len(args) == 2
    (mode, data), num = parse_rhs(args[1], identifiers)
    args[1] = (mode, data)

    stream.append((opcode, args))

    if mode == Mode.NEXT_TRYTE or mode == Mode.PTR_OFFSET:
        stream.append(Tryte.from_int(num))


def assemble_coroutine(instructions):
    section_pattern = r"^\.(\w+)\s*([\s\S]+?)(?=^\.|\Z)"
    sections = re.finditer(section_pattern, instructions, re.M)
    sections = {section.group(1): section.group(2) for section in sections}

    assert "data" in sections and "code" in sections

    data, identifiers = parse_data(sections["data"])
    instructions = sections["code"]

    labels = re.findall(r"(@.*)\n", instructions)

    identifiers |= {k: -999 for k in labels}
    instructions = instructions.strip().split("\n")
    instructions = (i.strip().split(";", maxsplit=1)[0] for i in instructions)
    instructions = list(instructions)

    stream = []
    for opcode in instructions:
        parse_one(stream, identifiers, opcode)

    stream = []
    for opcode in instructions:
        parse_one(stream, identifiers, opcode)

    stream = [ft.reduce(lambda t0, t1: Trits(t0.trits + t1.trits), mit.collapse(i)) for i in stream]
    stream = map(Tryte, (i.trits for i in stream))
    stream = list(stream)
    return data, stream


def parse_data(data):
    lines = data.splitlines()
    lines = map(str.strip, lines)
    lines = filter(None, lines)

    data = []
    pointers = {}

    for line in lines:
        name, result = line.split("=")
        name = name.strip()
        result = result.strip()

        pointers[name] = len(data)

        if result.startswith('"'):
            result = eval(result) + "\0"
            data += list(Tryte.from_int(terscii.to_terscii(i)) for i in result)
            continue

        if (num := parse_integer(result)) is not None:
            data += [Tryte.from_int(num)]
            continue

        if result.startswith("[") and result.endswith("]"):
            items = result[1:-1].split()
            items = list(map(parse_integer, items))
            assert all(map(lambda i: i is not None, items))
            data += list(map(Tryte.from_int, items))
            continue

        raise NotImplementedError()

    return data, pointers


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a TASM file.")
    parser.add_argument("file", metavar="F", type=p.Path, help="the file to read data from")

    args = parser.parse_args()

    with args.file.open("r", encoding="utf-8") as file:
        contents = file.read()

    data, program = assemble_coroutine(contents)

    vm = VirtualMachine(program)
    vm.memory[: len(data)] = data

    vm.run()
    output_info()
    output_info("===| Program |===\n")
    output_info(hept_dump(vm.program))
    output_info("===| Registers |===\n")
    output_info(hept_dump(vm.registers))
    output_info("===| Memory |===\n")
    output_info(hept_dump(vm.memory[:81]))
    output_info("===| Output buffer |===\n")
    output_info(vm.buffer)


if __name__ == "__main__":
    main()
