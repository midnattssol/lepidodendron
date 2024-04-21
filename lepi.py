#!/usr/bin/env python3.11
""""""
from __future__ import annotations
from parsing import *
import argparse


def parse_to_vm(contents):
    result = parse_file(MetadataString.from_str(contents))

    if result.is_err:
        return result

    program, ram_state = result.unwrap()

    vm = VirtualMachine(program)
    vm.memory[: len(ram_state)] = ram_state
    vm.set_register(Register.SP, Tryte.from_int(len(ram_state)))
    return Ok(vm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a TASM file.")
    parser.add_argument("file", metavar="F", type=p.Path, help="the file to read data from")
    parser.add_argument("--debug", action="store_true", help="show debug information")

    args = parser.parse_args()

    with args.file.open("r", encoding="utf-8") as file:
        contents = file.read()

    result = parse_to_vm(contents)

    if result.is_err:
        for item in result.unwrap_err():
            print(item)
        exit(1)

    vm = result.unwrap()

    max_memdump = 81
    if not args.debug:
        vm.log_execution = lambda *args, **kwargs: None
    vm.run()

    if args.debug:
        print()
        print("===| Program |===\n")
        print(hept_dump(vm.program))
        print("===| Registers |===\n")
        print(hept_dump(vm.registers))
        print("===| Memory |===\n")
        print(hept_dump(vm.memory[:max_memdump]))
        print("===| Output buffer |===\n")
        print(vm.buffer)

    else:
        print(vm.buffer)


if __name__ == "__main__":
    main()
