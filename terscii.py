#!/usr/bin/env python3.11
"""Convert from and to TerSCII."""
import string


def is_printable(n):
    return (0 <= n < len(MAPPING)) and len(MAPPING[n]) == 1


def is_whitespace(n):
    return from_terscii(n) in string.whitespace + "\0"


def from_terscii(n):
    return MAPPING[n]


def to_terscii(n):
    return MAPPING.index(n)


MAPPING = [
    "\0",
    "\n",
    "ET",
    "LR",
    "OP",
    "RL",
    "SU",
    "\t",
    "SD",
    " ",
    "-",
    "'",
    ",",
    ";",
    ":",
    ".",
    "!",
    "?",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "_",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
