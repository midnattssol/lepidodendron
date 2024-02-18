#!/usr/bin/env python3.11
""""""
from __future__ import annotations
import array
import copy as cp
import dataclasses as dc
import functools as ft
import itertools as it
import operator as op
import string
import typing as t

import more_itertools as mit
import terscii
from ternary import *
from agave import *

NumLike = t.Union[int, str]
T = t.TypeVar("T")
This, Self = t.TypeVar("This"), t.TypeVar("Self")

# ===| Classes |===


class Mode:
    """A read mode for the machine."""

    _mode_counter = count_trits(2)

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

    _opcode_counter = count_trits(3)

    # Memory
    STORE = next(_opcode_counter)
    LOAD = next(_opcode_counter)
    LOADNZ = next(_opcode_counter)

    # Arithmetic operations
    ADD = next(_opcode_counter)
    MUL = next(_opcode_counter)
    DIV = next(_opcode_counter)
    SUBB = next(_opcode_counter)
    ADDC = next(_opcode_counter)
    CMP = next(_opcode_counter)

    # Trit twiddling
    TSUM = next(_opcode_counter)
    ROT = next(_opcode_counter)
    TRITMUL = next(_opcode_counter)
    TRITEQ = next(_opcode_counter)

    # I/O
    INPUT = next(_opcode_counter)
    OUTPUT = next(_opcode_counter)

    NOOP = next(_opcode_counter)


class Register:
    """A mapping between register names and their indices."""

    _register_counter = count_trits(2)

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

    def log_execution(self, string):
        print(string, end="")

    def run(self):
        """Run the program currently stored in memory."""
        n_instrs = 0
        self.log_execution("Running program...\n")

        while 0 <= (current_instr_index := self.get_register(Register.PC).as_int()) < len(self.program):
            n_instrs += 1
            current_instr = self.program[current_instr_index]
            self.execute_instruction(current_instr)

        self.log_execution(f"\nProgram execution finished after performing {n_instrs} instructions.")

    def get_ram(self, address: Tryte):
        try:
            return self._get_ram(address)
        except IndexError:
            print()
            print("segfault")
            exit(1)

    def _get_ram(self, address: Tryte):
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
        """Set the value of a register."""
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

        program_counter = self.registers[Register.PC.as_int()].as_int()

        self.log_execution(
            f"{hept_dump(self.registers).strip('0').strip()} | {get_name(Opcode, opcode).ljust(10)} <{tryte}>"
        )

        if opcode == Opcode.ADD:
            left = self.get_ram(address).as_int()
            right = self.get_register(register).as_int()

            result = left + right

            self.set_flag_on_overflow(result)
            self.set_register(register, Tryte.from_int(result))

        elif opcode == Opcode.MUL:
            left = self.get_ram(address).as_int()
            right = self.get_register(register).as_int()

            result = left * right

            self.set_flag_on_overflow(result)
            self.set_register(register, Tryte.from_int(result))

        elif opcode == Opcode.TRITEQ:
            self.set_register(
                register, Tryte(map(op.eq, self.get_register(register).trits, self.get_ram(address).trits))
            )

        elif opcode == Opcode.TRITMUL:
            self.set_register(
                register, Tryte(map(op.mul, self.get_register(register).trits, self.get_ram(address).trits))
            )

        elif opcode == Opcode.CMP:
            left = self.get_register(register).as_int()
            right = self.get_ram(address).as_int()
            self.registers[Register.FL.as_int() % 9].trits[-1] = sign(left - right)

        elif opcode == Opcode.OUTPUT:
            self.buffer += terscii.from_terscii(self.get_ram(address).as_int())

        elif opcode == Opcode.LOAD or opcode == Opcode.LOADNZ:
            # This has to be out here since it has side effects, and these should execute
            # even if the load is not performed.
            value = self.get_ram(address)

            should_execute = not (opcode == Opcode.LOADNZ and self.get_register(Register.FL).as_int() == 0)

            if should_execute:
                self.set_register(register, value)
                self.log_execution(f"    Loading {value.as_int()} <{value}> -> {get_name(Register, register)}")

                if register == Register.PC:
                    self.log_execution("\n")
                    return

        elif opcode == Opcode.STORE:
            pointer = self.get_register(register).as_int()
            address_value = self.get_ram(address).clone()
            self.log_execution(
                f"    Storing {address_value.as_int()} <{address_value}> -> {pointer}",
            )
            self.memory[pointer] = address_value

        else:
            raise NotImplementedError(f"Opcode {get_name(Opcode, opcode)} not implemented")

        # Advance the program counter.
        self.log_execution("\n")
        self.registers[Register.PC.as_int()].incr()


# ===| Functions |===


def get_name(cls, name):
    for attr in filter(str.isupper, dir(cls)):
        value = getattr(cls, attr)
        if value == name:
            return attr

    raise ValueError()


def prettyprint_enum(cls):
    iterator = [item for item in dir(cls) if item.isupper()]
    iterator = [[item, getattr(cls, item), f"{getattr(cls, item).as_int():+}"] for item in iterator]
    return tabulate(iterator)
