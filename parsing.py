#!/usr/bin/env python3.11
""""""
from __future__ import annotations
from parser_base import *
import regex as re
import dataclasses as dc

from agave import *
import itertools as it
import more_itertools as mit

import enum
import collections as col
from ternary import *
from vm import *
import functools as ft
import argparse
import pathlib as p


class TokenType(enum.Enum):
    SECTION = enum.auto()
    ASSIGN = enum.auto()
    INSTRUCTION = enum.auto()
    LABEL = enum.auto()


@dc.dataclass
class Token:
    token_type: TokenType
    data: str
    second_tryte: list = dc.field(default_factory=list)

    def map(self, func):
        return dc.replace(self, data=func(self.data))


@dc.dataclass
class ParsingErrData:
    source_file: ...
    sources: ...
    message: ...

    def __repr__(self):
        return f"<{type(self).__qualname__} {self.sources!r} {self.message!r}>"


# ===| Parsing |===


def parse_file(tokens):
    source_file = tokens
    lines = tokens.findall_on(r"[^\n]+")
    lines = [mit.first(i.split(BEGIN_COMMENT)).strip() for i in lines]
    lines = filter(None, lines)
    lines = map(parse_line, lines)

    out = Result.lift_iter(lines).bind(ft.partial(compile_tokens, source_file))

    return out


def parse_line(string):
    out = string.tagged_regex(
        {
            TokenType.ASSIGN: rf"{IDENTIFIER_REGEX}\s*=.*",
            TokenType.INSTRUCTION: rf"{OPNAMES_REGEX}.*",
            TokenType.LABEL: rf"@{IDENTIFIER_REGEX}",
            TokenType.SECTION: rf"\.{IDENTIFIER_REGEX}",
        }
    )

    if out.is_nil:
        return Err(f"Unparseable line {string}")

    tag, _ = out.unwrap()
    return Ok(Token(tag, string))


# ===| Error handling |===


def compile_tokens(source_file, tokens) -> [Err]:
    """Find possible errors in the tokens."""
    result = compute_sections(source_file, tokens)
    memory_variables_and_size = result.bind(
        lambda sections: allocate_memory(source_file, sections[".data"]) if ".data" in sections else Ok({})
    )

    if memory_variables_and_size.is_err:
        return memory_variables_and_size

    sections = result.unwrap()
    memory_identifiers = memory_variables_and_size.unwrap()

    ram_state = list(mit.flatten(i[2] for i in memory_identifiers.values()))

    label_names = list(filter(lambda token: token.token_type == TokenType.LABEL, tokens))
    label_names = [i.data.string for i in label_names]

    executable_stream = make_stream(
        source_file,
        sections[".code"],
        set(memory_identifiers) | set(label_names),
        {k: Tryte.from_int(v[1]) for k, v in memory_identifiers.items()},
    )

    return executable_stream.map(lambda x: (x, ram_state))


def is_section(token):
    return token.token_type == TokenType.SECTION


def compute_section_names_and_ranges(source_file, tokens):
    markers = list(filter(is_section, tokens))
    names = (token.data.string for token in markers)
    ranges = (token.data.metadata.start for token in markers)
    ranges = mit.pairwise((*ranges, len(source_file)))
    ranges = it.starmap(range, ranges)

    return names, ranges


def compute_sections(source_file: MetadataString, tokens: [Token]) -> Result:
    """Divide the code into sections and verify that all necessary sections are defined."""

    by_section_name = mit.map_reduce(filter(is_section, tokens), lambda token: token.data.string)

    duplicate_keys = {k: v for k, v in by_section_name.items() if len(v) > 1}
    unrecognized_sections = {k: v for k, v in by_section_name.items() if k not in ALLOWED_SECTIONS}
    missing_sections = set(REQUIRED_SECTIONS) - set(by_section_name)

    errors = []

    for name, bad_tokens in duplicate_keys.items():
        err = make_err_from(
            source_file,
            bad_tokens,
            f"A section may only be defined once, but section `{name}` was defined multiple times.",
        )
        errors.append(err)

    for name, bad_tokens in unrecognized_sections.items():
        err = make_err_from(source_file, bad_tokens, f"Section `{name}` is not a recognized section.")
        errors.append(err)

    for missing_section in missing_sections:
        err = make_err_from(source_file, [], f"Section `{missing_section}` is required, but is never defined.")
        errors.append(err)

    out = Result.lift_iter(errors)

    if out.is_err:
        return out

    section_names, section_ranges = compute_section_names_and_ranges(source_file, tokens)

    spans = dict(zip(section_names, section_ranges))

    errors = []

    for key, token_types in SECTION_LOCKED_TOKEN_TYPES.items():
        bad_statements = [
            i for i in tokens if i.token_type in token_types and not (key in spans and token_in_span(i, spans[key]))
        ]

        for bad_statement in bad_statements:
            type_name = bad_statement.token_type.name.lower()
            err = Err(f"Statement {bad_statement} of type `{type_name}` outside `{key}` section.")
            errors.append(err)

    output = list(mit.split_at(tokens, is_section))[1:]
    sections = dict(zip(spans, output, strict=True))

    return Result.lift_iter(errors).map(const(sections))


# ===| Pre-compiled memory section |===


def parse_assignment_right_hand_side(rhs: str) -> Result[List[Tryte], str]:
    """Attempt to parse a right-hand side of an assignment in a `.data` section."""

    if not rhs.endswith('"') == rhs.startswith('"'):
        return Err(f"Unparsable right-hand side of assignment ({rhs}) in source file.")

    constant_is_string = rhs.startswith('"')

    if constant_is_string:
        # TODO: better escapes
        rhs = rhs.removeprefix('"').removesuffix('"')
        rhs = rhs.replace("\\0", "\0")
        return Ok([Tryte.from_int(terscii.to_terscii(i)) for i in rhs.string])

    parsed_num = parse_one_tryte_integer(rhs)

    # The constant is a number.
    if parsed_num.is_some:
        return Ok([parsed_num.unwrap()])

    # The constant is a list of numbers.
    if rhs.startswith("[") and rhs.endswith("]"):
        tokens = rhs.removeprefix("[").removesuffix("]").split()
        tokens = map(parse_one_tryte_integer, tokens)
        result = Maybe.lift_iter(tokens)
        return result.map(Ok).unwrap_or(Err(f"Badly formed list {rhs} in assignment in source file."))

    return Err(f"Unparsable right-hand side of assignment ({rhs}) in source file.")


def allocate_memory(source_file, data_section):
    """Allocate the initial memory."""

    def parse_assignment(token):
        lhs, rhs = token.data.split("=")
        name = lhs.strip()
        value = rhs.strip()
        return parse_assignment_right_hand_side(value).map(lambda x: (token, name, x))

    pointers = {}
    assignments = Result.lift_iter(map(parse_assignment, data_section))

    if assignments.is_err:
        return assignments

    memory = assignments.unwrap()
    repeated_occurrences = {k: v for k, v in mit.map_reduce(memory, lambda x: x[1].string).items() if len(v) > 1}

    maybe_errs = []

    for key, values in repeated_occurrences.items():
        err_info = [i[0].data for i in values]
        err = make_err_from(source_file, err_info, f"Repeated occurrence of key `{key}` in assignments")
        maybe_errs.append(err)

    maybe_errs = Result.lift_iter(maybe_errs)

    if maybe_errs.is_err:
        return maybe_errs

    sources, variable_names, trytes = zip(*memory)
    trytes_bkp = trytes
    variable_names = [i.string for i in variable_names]
    trytes = it.accumulate([0, *map(len, trytes)])
    trytes = list(trytes)[:-1]

    variable_data = zip(variable_names, zip(sources, trytes, trytes_bkp))
    variable_data = dict(variable_data)
    return Ok(variable_data)


# ===| Executable code section |===


def parse_one_executable_line(source_file, token, identifiers):
    command, *args = token.data.split()
    args = [arg.removesuffix_re(r"\s*,$") for arg in args]

    expected_arity = get_arity(getattr(Opcode, command.string.upper()))
    actual_arity = len(args)

    if actual_arity != expected_arity:
        return make_err_from(
            source_file,
            [token],
            f"Command `{command}` has arity `{expected_arity}`, but recieved `{actual_arity}` argument(s).",
        )
        raise NotImplementedError()

    lhs, rhs = padded_left(map(Just, args), Nil, 2)

    lhs = lhs.map(lambda lhs: parse_register(source_file, lhs)).unwrap_or(Ok([Trits.zero(2)]))
    rhs = rhs.map(lambda rhs: parse_rhs(source_file, rhs, identifiers)).unwrap_or(Ok([Trits.zero(4)]))

    val = getattr(Opcode, command.string.upper())
    args = [Ok([val]), lhs, rhs]
    args = Result.lift_iter(args)

    if not args:
        return args

    args = list(mit.flatten(args.unwrap()))

    # HACK: pass the string last if there is one and dont concat it - really needs a rework!
    last = []
    if isinstance(args[-1], MetadataString):
        last = [args.pop().string]

    args = ft.reduce(Trits.concat, args)
    args = map(Tryte, mit.chunked(args.trits, 9))
    return Ok(list(args) + last)


def make_stream(source_file, tokens, identifiers, labels):
    trytecode: (str | Tryte) = []
    labels = labels.copy()

    for token in tokens:
        if token.token_type == TokenType.INSTRUCTION:
            out = parse_one_executable_line(source_file, token, identifiers)

            if out.is_err:
                return out

            trytecode.extend(out.unwrap())
            continue

        if token.token_type == TokenType.LABEL:
            labels[token.data.string] = Tryte.from_int(len(trytecode))
            continue

        raise Never

    trytecode = [labels[x] if isinstance(x, str) else x for x in trytecode]
    return Ok(trytecode)


def parse_register(source_file, lhs) -> Result(Trits[2]):
    lhs_debug = lhs
    lhs = lhs.string
    normalized_name = lhs.upper()

    if not hasattr(Register, normalized_name):
        return make_err_from(source_file, [lhs_debug], f"Expected a register at position, but got `{lhs}`.")

    return Ok([getattr(Register, normalized_name)])


def parse_rhs(source_file, rhs, identifiers) -> Result(Trits[4, 2] | Trits[4, 2, 9]):
    """Returns the 4-trit fragment and a Maybe representing the next bit."""
    if rhs.string in identifiers:
        return Ok([Mode.NEXT_TRYTE, Trits.zero(2), rhs])

    elif is_label(rhs):
        # TODO: "did you mean x"
        return make_err_from(source_file, [rhs], f"Label `{rhs}` is never defined.")

    if (num_tryte := parse_one_tryte_integer(rhs)).is_some:
        num_tryte = num_tryte.unwrap()
        num = num_tryte.as_int()

        # Store small values in the last bits.
        if fits_into_n_trits(num, 2):
            return Ok([Mode.IMMEDIATE, Trits.trits_from_int(num, 2)])

        # If it's a big value, the next tryte is allocated to it.
        if fits_into_n_trits(num, 9):
            return Ok([Mode.NEXT_TRYTE, Trits.zero(2), num_tryte])

        # If it would cause an overflow, we report an error.
        return make_err_from(
            source_file,
            f"Immediate value {rhs} ({num}) does not fit into a signed tryte and would cause an overflow.",
            [rhs],
        )

    is_pointer = rhs.startswith("*")

    if not is_pointer:
        return parse_register(source_file, rhs).map(lambda x: [Mode.REGISTER] + x)

    is_pointer_offset = "->" in rhs

    if is_pointer_offset:
        # BUG: what if multiple arrows?
        rhs, offset_str = rhs.split("->")
        offset = parse_one_tryte_integer(source_file, offset_str)
        register = parse_register(source_file, rhs)

        if offset.is_err:
            return make_err_from(
                source_file, f"Expected numeric pointer offset at position, but got `{offset_str}`", [offset_str]
            )

        if register.is_err:
            return register

        return Ok([Mode.PTR_OFFSET, *register.unwrap(), offset.unwrap()])

    # The value is an indirect value.
    # Check if the value is some kind of incrementing/decrementing operation,
    # or otherwise pass it as a simple pointer.
    rhs = rhs.removeprefix("*")

    result = rhs.tagged_regex(
        [
            (Mode.PTR_POSTDECREMENT, r"^(.*)(?=--$)"),
            (Mode.PTR_POSTINCREMENT, r"^(.*)(?=\+\+$)"),
            (Mode.PTR_PREINCREMENT, r"(?<=^\+\+)(.*)$"),
            (Mode.PTR_PREDECREMENT, r"(?<=^--)(.*)$"),
            (Mode.PTR, r"^(.*)$"),
        ]
    )

    mode, register = result.unwrap()
    rhs = parse_register(source_file, register).map(lambda x: [mode] + x)
    return rhs


# ===| Utilities |===


def token_in_span(token, span):
    return token.data.metadata.start in span and token.data.metadata.stop in span


def make_err_from(source_file, bad_tokens, message):
    assert isinstance(source_file, MetadataString)
    assert isinstance(bad_tokens, (list, tuple))
    assert isinstance(message, str)
    return Err(ParsingErrData(source_file, bad_tokens, message))


def is_label(string):
    return string.startswith("@")


def parse_one_tryte_integer(input_str: str) -> Maybe[Tryte]:
    if isinstance(input_str, MetadataString):
        input_str = input_str.string

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
            return Just(Tryte.from_int(sign * int(match.group(1), v)))

    # Handle balanced number systems on its own.
    if match := re.match(r"^0t([01T]+)$", input_str):
        return Just(Tryte.from_int(sign * Tryte.from_balanced_ternary(match.group(1)).as_int()))

    # Handle balanced heptavintimal.
    if match := re.match(r"^0h([0-9a-zA-Z]+)$", input_str):
        return Just(Tryte.from_int(sign * Tryte.from_balanced_heptavintimal(match.group(1)).as_int()))

    return Nil


def get_arity(opcode: Opcode) -> int:
    """Get the arity of an operation."""

    if opcode == Opcode.NOOP:
        return 0
    if opcode == Opcode.OUTPUT:
        return 1
    return 2


OPNAMES = [i.lower() for i in dir(Opcode) if i.isupper()]
# names = [x: getattr(Opcode, x) for x in names]

OPNAMES_REGEX = "|".join(map(re.escape, OPNAMES))
IDENTIFIER_REGEX = r"\w[\w\d_-]*"
BEGIN_COMMENT = ";"

ALLOWED_SECTIONS = {".data", ".code"}
REQUIRED_SECTIONS = {".code"}


SECTION_LOCKED_TOKEN_TYPES = {
    ".data": [TokenType.ASSIGN],
    ".code": [TokenType.LABEL, TokenType.INSTRUCTION],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a TASM file.")
    parser.add_argument("file", metavar="F", type=p.Path, help="the file to read data from")

    args = parser.parse_args()

    with args.file.open("r", encoding="utf-8") as file:
        contents = file.read()

    # data, program = assemble_program(contents)
    result = parse_file(MetadataString.from_str(contents))

    if result.is_err:
        for item in result.unwrap_err():
            print(item)
        exit(1)

    program, ram_state = result.unwrap()

    vm = VirtualMachine(program)
    vm.memory[: len(ram_state)] = ram_state
    vm.set_register(Register.SP, Tryte.from_int(len(ram_state)))

    # Set the stack pointer

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


# def main():
#     tokens = """
#     .data
#        a = 10    ; hello
#        foo = [1 2 3 4]
#        bar = "hello world!"
#        ; a = 999
#
#     .code
#        noop
#        loadnz r0, 999
#        loadnz pc, @hello
#        ; cmp r0, a
#        @hello
#        cmp r1, *r0
#        ; jnz 11
#     """
#
#     tokens = MetadataString.from_str(tokens)
#
#     result = parse_file(tokens)
#     print(result)

if __name__ == "__main__":
    main()
