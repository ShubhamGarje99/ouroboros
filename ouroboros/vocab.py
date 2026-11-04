"""Token layout shared by the text task and the self-printing task.

  [0, C)                 text characters
  [C, C+2k+1)            value tokens  V(q) for q in -k..k
  next 64                scale tokens  S(i)
  next 32                row-address digits (base 32, two digits per row)
  next 3                 <Q> <END> <SCALE>
  next n_mat             <MAT_i>, one per matrix in the registry
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import N_SCALE_LEVELS, ROW_BASE


@dataclass
class Vocab:
    chars: str
    k: int
    mat_names: list[str]

    def __post_init__(self):
        self.C = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.VAL0 = self.C
        self.SCALE0 = self.VAL0 + 2 * self.k + 1
        self.DIG0 = self.SCALE0 + N_SCALE_LEVELS
        self.Q = self.DIG0 + ROW_BASE
        self.END = self.Q + 1
        self.SCALE_ADDR = self.Q + 2
        self.MAT0 = self.Q + 3
        self.size = self.MAT0 + len(self.mat_names)
        self.mat_index = {n: i for i, n in enumerate(self.mat_names)}
        assert self.size <= ROW_BASE ** 2, "too many tokens for two base-32 row digits"

    # text
    def encode_text(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode_text(self, ids) -> str:
        return "".join(self.chars[i] if i < self.C else f"<{i}>" for i in ids)

    # printing
    def val(self, q: int) -> int:
        return self.VAL0 + q + self.k

    def unval(self, t: int) -> int | None:
        q = t - self.VAL0 - self.k
        return q if -self.k <= q <= self.k else None

    def scale_tok(self, i: int) -> int:
        return self.SCALE0 + i

    def unscale(self, t: int) -> int | None:
        i = t - self.SCALE0
        return i if 0 <= i < N_SCALE_LEVELS else None

    def row_prefix(self, mat: str, row: int) -> list[int]:
        hi, lo = divmod(row, ROW_BASE)
        return [self.Q, self.MAT0 + self.mat_index[mat], self.DIG0 + hi, self.DIG0 + lo]

    def scale_prefix(self, mat: str) -> list[int]:
        return [self.Q, self.MAT0 + self.mat_index[mat], self.SCALE_ADDR]

    PREFIX_LEN = 4  # row prefix length; values start here
