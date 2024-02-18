#!/usr/bin/env python3.11
"""Convert from and to TerSCII.

# Sources

- Jones, W. Douglas. [TerSCII](https://homepage.cs.uiowa.edu/~dwjones/ternary/terscii.shtml)
"""
import string


def is_printable(n):
    unprintable = {"\u0003", "\u200e", "OP", "\u200f", "SU", "\t", "SD"}
    return (0 <= n < len(TERSCII_TO_ASCII)) and TERSCII_TO_ASCII[n] not in unprintable


def is_whitespace_or_null(n):
    return from_terscii(n) in string.whitespace + "\0"


def from_terscii(n):
    return TERSCII_TO_ASCII[n]


def to_terscii(n):
    return TERSCII_TO_ASCII.index(n)


# Code      Meaning
#
# ES        End of String, analogous to NULL
# EL        End of Line, analogous to LF or CR/LF
# ET        End of Text file
# LR        Left to Right rendering of following text
# OP        OverPrint following text on previous char
# RL        Right to Left rendering of following text
# SU        Shift Up (superscript) following by 1/3 baseline
# HT        Horizontal Tab in current rendering direction
# SD        Shift Down (subscript) following by 1/3 baseline
# SP        Space


TERSCII_TO_ASCII = [
    "\0",
    "\n",
    "\u0003",
    "\u200e",
    "OP",  # Placeholder name for Overprint
    "\u200f",
    "SU",  # Placeholder name for Shift Up
    "\t",
    "SD",  # Placeholder name for Shift Down
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
