#!/usr/bin/env python3.11
"""Parser utilities.

"""
from __future__ import annotations

import dataclasses as dc

import regex as re
from agave import *
import itertools as it
import more_itertools as mit

__all__ = ["Parser", "MetadataString", "Metadata"]


@dc.dataclass
class Parser:
    """Parse an expression."""

    input_expr: MetadataString
    index: int = 0

    # ===| Properties |===

    @property
    def exhausted(self):
        return self.index >= len(self.input_expr)

    @property
    def onwards(self):
        return self.input_expr[self.index :]

    # ===| Peeking |===

    def peek(self, n=1):
        return self.onwards[:n]

    def skip(self, n=1):
        self.index += n

    # ===| Unconditional moving |===

    def take_until(self, cond: str | callable) -> str:
        if isinstance(cond, str):
            return self.take_until_re(cond)

        out = []
        while (not self.exhausted) and not cond(self.input_expr[self.index]):
            char = self.input_expr[self.index]
            self.index += 1
            out.append(char)

        return "".join(out).strip()

    def take_until_re(self, data):
        result = re.search(data, self.input_expr.string, re.M, pos=self.index)
        last = result.start() if result else len(self.input_expr)

        prev_index = self.index
        self.index = last
        return self.input_expr[prev_index : self.index]

    # ===| Conditional moving |===

    def try_take_raw(self, data):
        return self.try_take(re.escape(data))

    def try_take(self, data):
        if match := re.match(data, self.onwards.string):
            prev = self.index
            self.index += match.end()
            return Just(self.input_expr[prev : self.index])

        return Nil

    def err_from_source(self, position, message):
        return Err((message, Just(position.metadata)))

    def err_general(self, message):
        return Err((message, Nil))

    def err_from_position(self, message):
        return Err((message, Metadata(self.index, self.index)))


# ===| Utilities |===


def on_str(funcname: str) -> Fn[Token]:
    """Apply a function to the base string."""

    def inner(token, *a, **kw):
        result = getattr(token.string, funcname)(*a, **kw)
        return result

    inner.__name__ = funcname
    return inner


def and_lift(funcname: str, direct=False) -> Fn[Token]:
    """Apply a function to the base string and add this item's metadata to it."""

    def inner(token, *a, **kw):
        result = getattr(token.string, funcname)(*a, **kw)
        result = token.lift(result)
        if (not direct) and result.metadata.direct:
            result.metadata = dc.replace(result.metadata, direct=False)
        return result

    inner.__name__ = funcname
    return inner


def get_slice_indices(slicer, list_length):
    start = slicer.start if slicer.start is not None else 0
    stop = slicer.stop if slicer.stop is not None else list_length

    assert slicer.step == 1 or slicer.step is None

    # Adjust negative indices & bound stop
    start = max(start, 0) if start >= 0 else max(list_length + start, 0)
    stop = max(stop, 0) if stop >= 0 else max(list_length + stop, 0)
    stop = min(stop, list_length)

    # Calculate the indices
    indices = [start, stop]
    return indices


@dc.dataclass(slots=True)
class MetadataString:
    string: str
    metadata: Metadata

    def lift(self, item: str) -> MetadataString:
        return dc.replace(self, string=item)

    # ===| Dunders |===

    def __repr__(self):
        return str(self)

    def __str__(self):
        indirectly = "directly" if self.metadata.direct else "indirectly"
        return f"<{type(self).__qualname__} {self.string!r} derived {indirectly} from {self.metadata.start}:{self.metadata.stop}>"

    def __iter__(self):
        yield from map(self.lift, self.string)

    def __getitem__(self, key):
        if isinstance(key, slice):
            assert key.step == 1 or key.step is None

            if self.metadata.direct:
                result = self.string[key]
                start, stop = get_slice_indices(key, len(self))
                out_metadata = dc.replace(
                    self.metadata,
                    start=self.metadata.start + start,
                    stop=self.metadata.stop - (len(self) - stop),
                    direct=True,
                )

                if out_metadata.start < out_metadata.stop:
                    result = MetadataString(result, out_metadata)
                    return result

                result = MetadataString(result, dc.replace(self.metadata, direct=True))
                return result

            return self.lift(self.string[key])

        return self.string[key]

    def __len__(self):
        return len(self.string)

    # ===| Regexes |===

    def tagged_regex(self, patterns):
        if isinstance(patterns, dict):
            patterns = patterns.items()

        for tag, pattern in patterns:
            if match := self.search_first(pattern):
                return Just((tag, match.unwrap()))

        return Nil

    def match(self, pattern, *a, **kw):
        match = re.match(pattern, self.string, *a, **kw)
        if match:
            return Just(self[slice(*match.span())])
        return Nil

    def search_first(self, pattern, *a, **kw):
        match = re.search(pattern, self.string, *a, **kw)
        if match:
            return Just(self[slice(*match.span())])
        return Nil

    def removesuffix_re(self, pattern):
        return self.search_first(pattern).map(len).map(lambda num: self[:-num]).unwrap_or(self)

    def findall_and_splits(self, pattern, *a, **kw):
        occurrences = list(re.finditer(pattern, self.string, *a, **kw))
        spans = mit.flatten(i.span() for i in occurrences)
        spans = mit.pairwise(spans)
        spans = it.starmap(slice, spans)
        spans = list(zip(spans, it.cycle([True, False])))

        if not spans:
            return [(self, False)]

        spans = [(slice(0, spans[0][0].start), False), *spans, (slice(spans[-1][0].stop, len(self.string)), False)]
        # spans = [i for i in spans if not (i[0].start == i[0].stop)]
        spans = [(self[i[0]], i[1]) for i in spans]

        return spans

    def findall_on(self, pattern, *a, **kw):
        return (i[0] for i in self.findall_and_splits(pattern) if i[1])

    def split(self, pattern=r"\s+"):
        return (i[0] for i in self.findall_and_splits(pattern) if not i[1])

    @classmethod
    def from_str(cls, string):
        return cls(string, Metadata(0, len(string), direct=True))

    # ===| Inherited |===

    startswith = on_str("startswith")
    endswith = on_str("endswith")

    count = on_str("count")
    find = on_str("find")
    index = on_str("index")
    isalnum = on_str("isalnum")
    isascii = on_str("isascii")
    isdecimal = on_str("isdecimal")
    isprintable = on_str("isprintable")
    isspace = on_str("isspace")
    isdigit = on_str("isdigit")
    isalpha = on_str("isalpha")
    isdigit = on_str("isdigit")
    islower = on_str("islower")
    isupper = on_str("isupper")
    isspace = on_str("isspace")

    replace = and_lift("replace")
    lower = and_lift("lower")
    upper = and_lift("upper")
    title = and_lift("title")
    capitalize = and_lift("capitalize")
    swapcase = and_lift("swapcase")

    def removeprefix(self, prefix):
        return self[len(prefix) :] if self.startswith(prefix) else self

    def removesuffix(self, suffix):
        return self[: -len(suffix)] if self.endswith(suffix) else self

    def lstrip(self, *a):
        result = self.string.lstrip(*a)
        diff = len(self.string) - len(result)
        out = self[diff:]
        return out

    def rstrip(self, *a):
        result = self.string.rstrip(*a)
        out = self[: len(result)]
        return out

    def strip(self, *a):
        return self.lstrip(*a).rstrip(*a)

    def __bool__(self):
        return bool(self.string)

    def __contains__(self, val):
        return val in self.string


@dc.dataclass(slots=True)
class Metadata:
    start: int
    stop: int
    direct: bool = False
