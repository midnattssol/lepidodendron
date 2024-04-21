#!/usr/bin/env python3.11
""""""
from __future__ import annotations

import collections as col
import copy as cp
import dataclasses as dc
import enum
import functools as ft
import itertools as it
import pathlib as p
import math

import more_itertools as mit
import regex as re
from agave import *
from parser_base import *
from ternary import *
from vm import *


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


@dc.dataclass
class MemoryIdentifierData:
    sources: ...
    trytes: ...
    pointer: ...


# ===| Parsing |===


def parse_file(source_file: MetadataString) -> Result[([Tryte], [Tryte]), any]:
    """Parse a file."""
    lines = source_file.findall_on(r"[^\n]+")
    lines = [mit.first(i.split(BEGIN_COMMENT)).strip() for i in lines]
    lines = [tokenize_line(line) for line in lines if line]
    lines = Result.lift_iter(lines)

    if lines.is_err:
        return lines

    tokens = lines.unwrap()
    result = compute_sections(source_file, tokens)

    if result.is_err:
        return result

    sections = result.unwrap()
    memory_variables_and_size = allocate_memory(source_file, sections[".data"]) if ".data" in sections else Ok({})

    if memory_variables_and_size.is_err:
        return memory_variables_and_size

    memory_identifiers = memory_variables_and_size.unwrap()

    ram_state = list(mit.flatten(i.trytes for i in memory_identifiers.values()))

    label_names = [token.data.string for token in tokens if token.token_type == TokenType.LABEL]
    label_lookup = {k: Tryte.from_int(v.pointer) for k, v in memory_identifiers.items()}

    executable_stream = compile_section_code(
        source_file,
        sections[".code"],
        set(memory_identifiers) | set(label_names),
        label_lookup,
    )

    return executable_stream.map(lambda x: (x, ram_state))


def tokenize_line(string: str) -> Result[Token, str]:
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


def is_token_section(token):
    return token.token_type == TokenType.SECTION


def is_token_instruction(token):
    return token.token_type == TokenType.INSTRUCTION


def compute_section_names_and_ranges(source_file, tokens):
    markers = list(filter(is_token_section, tokens))
    names = (token.data.string for token in markers)
    ranges = (token.data.metadata.start for token in markers)
    ranges = mit.pairwise((*ranges, len(source_file)))
    ranges = it.starmap(range, ranges)

    return names, ranges


def compute_sections(source_file: MetadataString, tokens: [Token]) -> Result:
    """Divide the code into sections and verify that all necessary sections are defined."""

    by_section_name = mit.map_reduce(filter(is_token_section, tokens), lambda token: token.data.string)

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

    output = list(mit.split_at(tokens, is_token_section))[1:]
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


def allocate_memory(source_file: MetadataString, data_section):
    """Allocate the initial memory."""

    def parse_assignment(token):
        lhs, rhs = token.data.split("=")
        name = lhs.strip()
        value = rhs.strip()
        out = parse_assignment_right_hand_side(value).map(lambda x: (token, name, x))
        return out

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
    variable_names = [i.string for i in variable_names]

    pointers = it.accumulate([0, *map(len, trytes)])
    pointers = list(pointers)[:-1]

    variable_data = zip(variable_names, map(MemoryIdentifierData, sources, trytes, pointers))
    variable_data = dict(variable_data)
    return Ok(variable_data)


# ===| Executable code section |===


ReferenceOrRawTrits, Reference, RawTrits = tagged_union(
    "ReferenceOrRawTrits", {"Reference": ["value"], "RawTrits": ["trit_obj"]}
)


def get_n_trites(expr):
    return expr.cases(lambda reference: 9, lambda trits: len(trits.trit_obj.trits))


def get_num_trytes(instructions):
    n_trits = sum(map(get_n_trites, mit.flatten(instructions)))
    return math.ceil(n_trits / 9)


def compile_section_code(source_file: MetadataString, tokens, identifiers, label_lookup):
    """Compile a token to trytecode."""

    def instr_index_to_tryte_index(index: int) -> int:
        return get_num_trytes(instructions[:index])

    def evenly_divisible_into_trytes(trytecode: Trits) -> bool:
        return len(trytecode) % 9 == 0

    def instruction_to_trits(instruction):
        items = (
            trits_or_label.cases(lambda label: label_lookup[label.value.string], lambda raw: raw.trit_obj).trits
            for trits_or_label in instruction
        )
        return list(mit.flatten(items))

    assert all(x.token_type in {TokenType.LABEL, TokenType.INSTRUCTION} for x in tokens)

    # Compile the instructions.
    instructions = filter(is_token_instruction, tokens)
    instructions = [compile_section_code_instr(source_file, token, identifiers) for token in instructions]
    instructions = Result.lift_iter(instructions)

    if instructions.is_err:
        return instructions

    instructions = instructions.unwrap()

    # Find the number of instructions before each instruction or label, and use this
    # to calculate the number of trytes before.
    non_labels = map(is_token_instruction, tokens)
    n_instrs_before = it.accumulate(non_labels, op.add, initial=0)

    indices_to_names = zip(tokens, n_instrs_before)
    indices_to_names = [(token, v) for token, v in indices_to_names if not is_token_instruction(token)]
    indices_to_names = {instr_index: token.data.string for token, instr_index in indices_to_names}
    indices_to_names = try_map_keys(instr_index_to_tryte_index, indices_to_names).unwrap()

    # Unwrap the singular locations once they've been verified.
    names_to_indices = try_flip_dict(indices_to_names)

    if names_to_indices.is_err:
        bad = names_to_indices.unwrap_err()
        result = [f"Label {k} was defined at {len(v)} different locations (indices {v})" for k, v in bad]
        return Err(result)

    label_lookup = label_lookup | map_values(Tryte.from_int, names_to_indices.unwrap())

    # Once we've finally converted all the labels, we can start
    # converting the instructions into trits.
    instructions = map(instruction_to_trits, instructions)
    instructions = list(instructions)

    # Make sure that each instruction is evenly divisible into some number of trytes.
    Result.expect_all(evenly_divisible_into_trytes, instructions).unwrap_or_raise()

    instructions = (mit.chunked(x, 9) for x in instructions)
    instructions = mit.flatten(instructions)
    trytecode = map(Tryte, instructions)
    return Ok(list(trytecode))


def compile_section_code_instr(source_file: MetadataString, token: Token, identifiers) -> Result[[Tryte], str]:
    command, *arguments = token.data.split()
    opcode_trits = getattr(Opcode, command.string.upper())

    arguments = [arg.removesuffix_re(r"\s*,$") for arg in arguments]

    expected_arity = get_arity(opcode_trits)
    actual_arity = len(arguments)

    if actual_arity != expected_arity:
        return make_err_from(
            source_file,
            [token],
            f"Command `{command}` has arity `{expected_arity}`, but recieved `{actual_arity}` argument(s).",
        )

    maybe_lhs, maybe_rhs = padded_left(map(Just, arguments), Nil, 2)

    maybe_lhs_trits = maybe_lhs.map(lambda lhs: parse_register(source_file, lhs)).unwrap_or(Ok([Trits.zero(2)]))
    maybe_rhs_trits = maybe_rhs.map(lambda rhs: parse_rhs(source_file, rhs, identifiers)).unwrap_or(Ok([Trits.zero(4)]))

    arguments = [Ok([opcode_trits]), maybe_lhs_trits, maybe_rhs_trits]
    arguments = Result.lift_iter(arguments)

    if arguments.is_err:
        return arguments

    arguments = list(mit.flatten(arguments.unwrap()))
    arguments = [Reference(x) if isinstance(x, MetadataString) else RawTrits(x) for x in arguments]

    return Ok(arguments)


def condense(arguments):
    trytes: [ReferenceOrRawTrits] = []

    for arg in arguments:
        if arg.is_reference():
            trytes.append(arg)
            continue

        if trytes and trytes[-1].is_raw_trits():
            trytes[-1].trit_obj.extend(arg.trit_obj)
            continue

        trytes.append(cp.deepcopy(arg))

    return trytes


def parse_register(source_file: MetadataString, lhs) -> Result(Trits[2]):
    lhs_debug = lhs
    lhs = lhs.string
    normalized_name = lhs.upper()

    if not hasattr(Register, normalized_name):
        return make_err_from(source_file, [lhs_debug], f"Expected a register at position, but got `{lhs}`.")

    return Ok([getattr(Register, normalized_name)])


def parse_rhs(source_file: MetadataString, rhs, identifiers) -> Result(Trits[4, 2] | Trits[4, 2, 9]):
    """Returns the 4-trit fragment and a Maybe representing the next bit."""
    if rhs.string in identifiers:
        return Ok([Mode.NEXT_TRYTE, Trits.zero(2), rhs])

    elif is_label(rhs):
        # TODO: "did you mean x"
        return make_err_from(source_file, [rhs], f"Label `{rhs}` is never defined.")

    num_tryte = parse_one_tryte_integer(rhs)
    is_string_literal = rhs.string.startswith("'") and rhs.string.endswith("'")

    if is_string_literal:
        num_tryte = terscii.to_terscii(rhs[1:-1].string)
        num_tryte = Just(Tryte.from_int(num_tryte))

    if num_tryte.is_some:
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


def is_label(string: str) -> bool:
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

    for pattern, base in number_literals.items():
        if match := re.match(pattern, input_str):
            return Just(Tryte.from_int(sign * int(match.group(1), base)))

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


REGISTERS = [i.lower() for i in dir(Register) if i.isupper()]

OPNAMES = [i.lower() for i in dir(Opcode) if i.isupper()]
OPNAMES_REGEX = "|".join(map(re.escape, OPNAMES))
IDENTIFIER_REGEX = r"\w[\w\d_-]*"
BEGIN_COMMENT = ";"

ALLOWED_SECTIONS = {".data", ".code"}
REQUIRED_SECTIONS = {".code"}

SECTION_LOCKED_TOKEN_TYPES = {
    ".data": [TokenType.ASSIGN],
    ".code": [TokenType.LABEL, TokenType.INSTRUCTION],
}
