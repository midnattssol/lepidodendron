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
from agave import *


@dc.dataclass
class Trits:
    trits: array

    def __post_init__(self):
        if not isinstance(self.trits, array.array):
            self.trits = array.array("b", map(int, self.trits))

    def extend(self, other):
        self.trits += other.trits
        return self

    @property
    def width(self):
        return len(self.trits)

    @classmethod
    def empty(cls):
        return cls.zero(0)

    @classmethod
    def zero(cls, limit):
        return cls.trits_from_int(0, limit)

    @classmethod
    def from_balanced_heptavintimal(cls, text):
        data = map(BALANCED_HEPTAVINTIMAL_DIGITS.index, text)
        data = (Trits.trits_from_int(i, 3) for i in data)
        result = ft.reduce(Trits.extend, data, Trits.empty())
        return pad_to_tryte(result).trits

    @classmethod
    def from_balanced_ternary(cls, text):
        text = "".join(text)
        text = map(lambda x: {"T": -1, "0": 0, "1": 1}[x], text)
        text = padded_left(text, 0, 9)
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
        return ternary_to_int(self.trits)

    def __eq__(self, other):
        assert len(self.trits) == len(other.trits), (self, other)
        return self.trits == other.trits

    def __repr__(self):
        return self.dump_ternary()

    def concat(self, other):
        out = cp.deepcopy(self)
        out.extend(other)
        return out


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

    def incr(self, diff: int = 1) -> Self:
        """Increment the number in place."""
        diff = diff.as_int() if isinstance(diff, Tryte) else diff
        self.trits = Tryte.from_int(self.as_int() + diff).trits

    def decr(self, diff: int = 1) -> Self:
        """Decrement the number in place."""
        self.incr(-diff)

    def dump_heptavintimal(self) -> str:
        chunks = mit.chunked(self.trits, 3)
        chunks = map(ternary_to_int, chunks)
        return "".join(BALANCED_HEPTAVINTIMAL_DIGITS[chunk] for chunk in chunks)

    def __eq__(self, other):
        return super().__eq__(other)

    def __repr__(self):
        return super().__repr__()


def base2base(num: NumLike, from_base: int, target_base: int, width: int = None) -> NumLike:
    """Convert between bases."""
    assert not (isinstance(num, int) and from_base != 10)
    assert from_base <= len(BALANCED_HEPTAVINTIMAL_DIGITS)

    num = num if isinstance(num, int) else base2int(num, from_base)

    # Make sure the item is within the max representable.
    if width is not None:
        max_representable = target_base**width
        num %= max_representable

    result = []

    while num:
        result.insert(0, BALANCED_HEPTAVINTIMAL_DIGITS[:target_base][num % target_base])
        num //= target_base

    result = "".join(result)
    zero_char = BALANCED_HEPTAVINTIMAL_DIGITS[0]
    min_width = width if width is not None else 1

    # Pad the result with zeroes if there is a target minimum width.
    return result.rjust(min_width, zero_char)


def base2int(num: NumLike, from_base: int) -> int:
    """Convert from a base to an integer."""
    powers = reversed(range(len(num)))
    digits = map(BALANCED_HEPTAVINTIMAL_DIGITS[:from_base].index, num)
    num = sum(map(lambda a, b: b * from_base**a, powers, digits))
    return num


def count_trits(width=9, mode="normal") -> t.Iterator[Trits]:
    """Generate trits.

    Modes: normal (0, 1, 2, 3, 4, ...) and interleave (0, 1, -1, 2, -2, ...).
    """
    if mode == "interleave":
        iterator = mit.interleave_longest(it.count(1, 1), it.count(-1, -1))
        iterator = it.chain((0,), iterator)
    elif mode == "normal":
        iterator = it.count(0)
    else:
        raise NotImplementedError(f"Unknown mode {mode!r}")

    iterator = it.islice(iterator, 3**width)
    yield from (Trits.trits_from_int(i, width) for i in iterator)


def ternary_to_int(num):
    return sum(v * 3**i for i, v in enumerate(num[::-1]))


def pad_to_tryte(iterator: Iterator[int]) -> Tryte:
    iterator = list(iterator)
    assert len(iterator) <= 9
    default_trit = 0
    return Tryte(padded_left(iterator, default_trit, 9))


def parse_tryte(expected, tryte):
    expected = list(expected)
    iterator = iter(tryte.trits)
    assert sum(expected) == len(tryte.trits)

    for num in expected:
        yield Trits(mit.take(num, iterator))


def hept_dump(mem) -> str:
    """Dump a heptavintimal representation of the entire program memory hexdump-style."""
    chunksize = 9
    chunks = mit.chunked(mem, chunksize)
    out = ""

    ptr = Tryte.from_int(0)
    sep = "   "

    default_repr_subchunk = "–" * len(Tryte.dump_heptavintimal(Tryte.from_int(0)))

    def heptavintimalize(subchunk):
        return " ".join(i.map(Tryte.dump_heptavintimal).unwrap_or(default_repr_subchunk) for i in subchunk)

    def terscii_or_default(i):
        num = i.as_int()
        default = "·"

        if not terscii.is_printable(num):
            return default

        result = terscii.from_terscii(num)
        return result if result == " " or not terscii.is_whitespace_or_null(num) else default

    for chunk in chunks:
        chunk = map(Just, chunk)
        chunk = mit.padded(chunk, Nil, 9)
        subchunks = list(mit.chunked(chunk, 3))

        chunk_str = sep.join(map(heptavintimalize, subchunks))
        chunk_as_terscii = "".join(i.map(terscii_or_default).unwrap_or("–") for subchunk in subchunks for i in subchunk)

        out += sep.join(
            [
                ptr.dump_heptavintimal().rjust(6, "0"),
                chunk_str,
                "|" + chunk_as_terscii + "|\n",
            ]
        )

        ptr = Tryte.from_int(ptr.as_int() + chunksize)

    return out.strip() + "\n"


def fits_into_n_trits(num: int, k: int) -> bool:
    """Find out if a number is representable in k-trit balanced ternary."""
    lower = Trits.from_balanced_ternary("T" * k)
    upper = Trits.from_balanced_ternary("1" * k)
    return upper.as_int() >= num >= lower.as_int()


# ===| Globals |===

BALANCED_HEPTAVINTIMAL_DIGITS = sorted(set(string.ascii_lowercase + string.digits) - {*"ijlyoqsuw"})
MEM_SIZE = 3**6
