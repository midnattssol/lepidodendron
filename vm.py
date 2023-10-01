#!/usr/bin/env python3.11
""""""
import array
import copy as cp
import dataclasses as dc
import itertools as it
import operator as op
import string
import typing as t

import more_itertools as mit
import terscii

output_info = print
# output_info = lambda *a, **kw: ...

NumLike = t.Union[int, str]
T = t.TypeVar("T")
This, Self = t.TypeVar("This"), t.TypeVar("Self")

# ===| Classes |===

_DIGITS = sorted(set(string.ascii_lowercase + string.digits) - {*"ijlyoqsuw"})
MEM_SIZE = 3**5


@dc.dataclass
class Trits:
    trits: array

    def __post_init__(self):
        if not isinstance(self.trits, array.array):
            self.trits = array.array("b", map(int, self.trits))

    @classmethod
    def zero(cls, limit):
        return cls.trits_from_int(0, limit)

    @classmethod
    def from_ternary(cls, text):
        text = "".join(text)
        text = map({"T": -1, "0": 0, "1": 1}.get, text)
        text = filter(lambda x: x is not None, text)
        text = list(text)
        text = [0] * (9 - len(text)) + text
        text = cls(text)
        return text

    @classmethod
    def trits_from_int(cls, num, limit):
        bias = base2int(["1"] * limit, 3)
        result = base2base(num + bias, 10, 3, limit)
        result = array.array("b", [int(i) - 1 for i in result])
        return cls(result)

    def clone(self):
        return cp.deepcopy(self)

    def dump_ternary(self):
        return "".join(map({1: "1", -1: "T", 0: "0"}.get, self.trits))

    def as_int(self):
        return btern2int(self.trits)

    def same_val(self, other):
        x = min(len(self), len(other)) - max(len(self), len(other))
        return self.trits[x:] == other.trits[x:]

    def __eq__(self, other):
        assert len(self.trits) == len(other.trits), (self, other)
        return self.trits == other.trits

    def __repr__(self):
        return self.dump_ternary()


@dc.dataclass
class Tryte(Trits):
    trits: array

    def __post_init__(self):
        super().__post_init__()
        assert len(self.trits) == 9

    @classmethod
    def from_int(cls, num: int) -> This:
        return cls.trits_from_int(num, 9)

    @classmethod
    def zero(cls) -> This:
        return cls(Trits.zero(9).trits)

    def incr(self, num: int = 1) -> Self:
        """Increment the number in place."""
        if isinstance(num, Tryte):
            num = num.as_int()
        self.trits = Tryte.from_int(self.as_int() + num).trits

    def decr(self, num: int = 1) -> Self:
        """Decrement the number in place."""
        self.incr(-num)

    def dump_heptavintimal(self) -> str:
        return "".join(str(_DIGITS[btern2int(trit)]) for trit in mit.chunked(self.trits, 3))

    def __eq__(self, other):
        return super().__eq__(other)

    def __repr__(self):
        return super().__repr__()


def base2base(num: NumLike, from_base: int, target_base: int, width: int = None) -> NumLike:
    """Convert between bases."""
    assert not (isinstance(num, int) and from_base != 10)
    assert from_base <= len(_DIGITS)

    num = num if isinstance(num, int) else base2int(num, from_base)

    # Make sure the item is within the max representable.
    if width is not None:
        max_representable = target_base**width
        num %= max_representable

    result = []

    while num:
        result.insert(0, _DIGITS[:target_base][num % target_base])
        num //= target_base

    result = "".join(result)
    zero_char = _DIGITS[0]
    min_width = width if width is not None else 1

    # Pad the result with zeroes if there is a target minimum width.
    return result.rjust(min_width, zero_char)


def base2int(num: NumLike, from_base: int) -> int:
    """Convert from a base to an integer."""
    powers = reversed(range(len(num)))
    digits = map(_DIGITS[:from_base].index, num)
    num = sum(map(lambda a, b: b * from_base**a, powers, digits))
    return num


def iota_trits(n_trits=9, mode="normal") -> t.Iterator[Trits]:
    """Generate trits.

    Modes: normal (0, 1, 2, 3, 4, ...) and interleave (0, 1, -1, 2, -2, ...).
    """
    if mode == "interleave":
        iterator = mit.interleave_longest(it.count(1, 1), it.count(-1, -1))
        iterator = it.chain((0,), iterator)
    elif mode == "normal":
        iterator = it.count(0)
    else:
        raise NotImplementedError(f"Mode {mode!r} not implemented")

    iterator = it.islice(iterator, 3**n_trits)
    yield from (Trits.trits_from_int(i, n_trits) for i in iterator)


class Mode:
    """A read mode for the machine."""

    _mode_counter = iota_trits(2)

    REGISTER = next(_mode_counter)
    IMMEDIATE = next(_mode_counter)
    NEXT_TRYTE = next(_mode_counter)
    PTR = next(_mode_counter)
    PTR_PREINCREMENT = next(_mode_counter)
    PTR_POSTINCREMENT = next(_mode_counter)
    PTR_PREDECREMENT = next(_mode_counter)
    PTR_POSTDECREMENT = next(_mode_counter)
    PTR_OFFSET = next(_mode_counter)


class Opcode:
    """A 3-trit code for an operation."""

    _opcode_counter = iota_trits(3)

    # Memory
    STORE = next(_opcode_counter)
    LOAD = next(_opcode_counter)
    LOADNZ = next(_opcode_counter)

    # Arithmetic operations
    ADD = next(_opcode_counter)
    SUB = next(_opcode_counter)
    MUL = next(_opcode_counter)
    DIV = next(_opcode_counter)
    SUBB = next(_opcode_counter)
    ADDC = next(_opcode_counter)
    CMP = next(_opcode_counter)

    # Trit twiddling
    TSUM = next(_opcode_counter)
    SHF = next(_opcode_counter)
    ROT = next(_opcode_counter)
    TRITMUL = next(_opcode_counter)
    TRITEQ = next(_opcode_counter)

    # I/O
    INPUT = next(_opcode_counter)
    OUTPUT = next(_opcode_counter)


class Register:
    """A mapping between register names and their indices."""

    _register_counter = iota_trits(2)

    R0 = next(_register_counter)
    R1 = next(_register_counter)
    R2 = next(_register_counter)
    R3 = next(_register_counter)
    R4 = next(_register_counter)
    R5 = next(_register_counter)
    FL = next(_register_counter)
    SP = next(_register_counter)
    PC = next(_register_counter)

    def __eq__(self, o):
        return self == o


@dc.dataclass
class VirtualMachine:
    """The virtual machine which executes the code."""

    program: list
    memory: list = dc.field(default_factory=lambda: [Tryte.zero() for _ in range(MEM_SIZE)])
    registers: list = dc.field(default_factory=lambda: [Tryte.zero() for _ in range(9)])

    buffer: str = ""

    def run(self):
        """Run the program currently stored in memory."""
        n_instrs = 0
        output_info("Running program...")

        while 0 <= (current_instr_index := self.get_register(Register.PC).as_int()) < len(self.program):
            n_instrs += 1
            current_instr = self.program[current_instr_index]
            self.execute_instruction(current_instr)

        output_info(f"\nProgram execution finished after performing {n_instrs} instructions.")

    def access_address(self, address: Tryte):
        """Access an address in a way which might have side effects."""
        mode, register = parse_tryte((2, 2), address)

        # Simply get the register.
        if mode == Mode.REGISTER:
            return self.get_register(register)

        # The data in the trits is the result.
        if mode == Mode.IMMEDIATE:
            return pad_to_tryte(register.trits)

        # The next tryte holds the result.
        if mode == Mode.NEXT_TRYTE:
            self.get_register(Register.PC).incr()
            return self.program[self.get_register(Register.PC).as_int()]

        # The register holds a pointer to the memory to read from.
        if mode == Mode.PTR:
            pointer_register = self.get_register(register)
            result = self.memory[pointer_register.as_int()]
            return result

        if mode == Mode.PTR_PREINCREMENT:
            pointer_register = self.get_register(register)
            pointer_register.incr()
            result = self.memory[pointer_register.as_int()]
            return result

        if mode == Mode.PTR_POSTINCREMENT:
            pointer_register = self.get_register(register)
            result = self.memory[pointer_register.as_int()]
            pointer_register.incr()
            return result

        if mode == Mode.PTR_PREDECREMENT:
            pointer_register = self.get_register(register)
            pointer_register.decr()
            result = self.memory[pointer_register.as_int()]
            return result

        if mode == Mode.PTR_POSTDECREMENT:
            pointer_register = self.get_register(register)
            result = self.memory[pointer_register.as_int()]
            pointer_register.decr()
            return result

        # The pointer is offset by some amount stored in the next trit.
        if mode == Mode.PTR_OFFSET:
            pointer_register = self.get_register(register)
            self.get_register(Register.PC).incr()
            offset = self.program[self.get_register(Register.PC).as_int()]
            result = self.memory[pointer_register.as_int() + offset.as_int()]
            return result

        raise NotImplementedError(mode)

    def set_register(self, register: Tryte, value: Tryte) -> None:
        """Set the value of a register-"""
        assert isinstance(value, Tryte)
        num = register.as_int() % 9
        self.registers[num] = value.clone()

    def get_register(self, register: Tryte) -> Tryte:
        """Get the value of a register."""
        num = register.as_int() % 9
        return self.registers[num]

    def set_flag_on_overflow(self, value: int) -> None:
        """Check overflow and set flags if necessary."""
        trit = 0
        if value > 3**9 // 2:
            trit = 1
        if value <= -(3**9 // 2):
            trit = -1

        # The overflow trit is the second from the back, the first being
        # the comparison trit.
        self.registers[Register.FL.as_int() % 9].trits[-2] = trit

    def execute_instruction(self, tryte):
        """Execute a single instruction."""
        opcode, register, address = parse_tryte((3, 2, 4), tryte)

        output_info(
            "pc =",
            self.registers[Register.PC.as_int()].as_int(),
            "|",
            get_name(Opcode, opcode).ljust(10),
            f"<{tryte}>",
            end="",
        )

        if opcode == Opcode.ADD:
            left = self.access_address(address).as_int()
            right = self.get_register(register).as_int()

            result = left + right

            self.set_flag_on_overflow(result)
            self.set_register(register, Tryte.from_int(result))

        elif opcode == Opcode.TRITEQ:
            self.set_register(
                register, Tryte(map(op.eq, self.get_register(register).trits, self.access_address(address).trits))
            )

        elif opcode == Opcode.TRITMUL:
            self.set_register(
                register, Tryte(map(op.mul, self.get_register(register).trits, self.access_address(address).trits))
            )

        elif opcode == Opcode.CMP:
            left = self.get_register(register).as_int()
            right = self.access_address(address).as_int()
            self.registers[Register.FL.as_int() % 9].trits[-1] = sign(left - right)

        elif opcode == Opcode.OUTPUT:
            self.buffer += terscii.from_terscii(self.access_address(address).as_int())

        elif opcode == Opcode.LOAD or opcode == Opcode.LOADNZ:
            # This has to be out here since it has side effects.
            value = self.access_address(address)

            if opcode != Opcode.LOADNZ or self.get_register(Register.FL).as_int() != 0:
                self.set_register(register, value)
                output_info(f"    Loading {value.as_int()} <{value}> -> {get_name(Register, register)}", end="")
                if register == Register.PC:
                    output_info()
                    return

        elif opcode == Opcode.STORE:
            pointer = self.get_register(register).as_int()
            address_value = self.access_address(address).clone()
            output_info(f"    Storing {address_value.as_int()} <{address_value}> -> {pointer}", end="")
            self.memory[pointer] = address_value

        else:
            raise NotImplementedError(f"Opcode {get_name(Opcode, opcode)} not implemented")

        # Advance the program counter.
        output_info()
        self.registers[Register.PC.as_int()].incr()


# ===| Functions |===


def get_name(cls, name):
    for attr in filter(str.isupper, dir(cls)):
        value = getattr(cls, attr)
        if value == name:
            return attr

    raise ValueError()


def tabulate(rows_and_columns, column_separator: str = "   ") -> str:
    """Format a list of lists into a table.

    All sublists must have the same length.

    Examples
    --------
    >>> tabulate([[10, "hello", "hi"], [9, "hi", "hii"]])
    10 hello hi
    9  hi    hii
    """
    rows_and_columns = list(map(list, rows_and_columns))

    if not rows_and_columns:
        return ""

    # Make sure that the table is rectangular.
    assert mit.all_equal(map(len, rows_and_columns)), rows_and_columns

    # Handle ANSI escape sequences gracefully.
    ljust = lambda x, width: x + " " * (width - len(x))

    # Turn all cells into strings.
    rows_and_columns = list(list(map(str, i)) for i in rows_and_columns)

    # Get the column-wise maximum length.
    transposed = zip(*rows_and_columns)
    transposed = list(transposed)

    max_len_by_row = [max(map(len, i)) for i in transposed]

    # Pad each row with whitespaces so they're as long as the max length of the row.
    # This ensures that the columns line up correctly.
    rows_and_columns = (map(ljust, i, max_len_by_row) for i in rows_and_columns)
    lines = map(column_separator.join, rows_and_columns)
    result = "\n".join(lines)

    return result


def prettyprint_enum(cls):
    iterator = [item for item in dir(cls) if item.isupper()]
    iterator = [[item, getattr(cls, item), f"{getattr(cls, item).as_int():+}"] for item in iterator]
    return tabulate(iterator)


def sign(x):
    return 0 if x == 0 else -1 if x < 0 else 1


def btern2int(num):
    return sum(v * 3**i for i, v in enumerate(num[::-1]))


def pad_to_tryte(iterator):
    iterator = list(iterator)
    assert len(iterator) <= 9
    return Tryte([0] * (9 - len(iterator)) + iterator)


def parse_tryte(expected, tryte):
    expected = list(expected)
    iterator = iter(tryte.trits)
    assert sum(expected) == len(tryte.trits)

    for num in expected:
        yield Trits(mit.take(num, iterator))
