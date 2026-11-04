from __future__ import annotations

import torch

from .model import GPT
from .vocab import Vocab


def load_text(path: str, split: float = 0.9):
    text = open(path, encoding="utf-8").read()
    chars = "".join(sorted(set(text)))
    n = int(len(text) * split)
    return text[:n], text[n:], chars


class TextBatches:
    def __init__(self, text: str, vocab: Vocab, seq_len: int, batch: int, gen: torch.Generator):
        self.ids = torch.tensor(vocab.encode_text(text), dtype=torch.int64)
        self.seq_len, self.batch, self.gen = seq_len, batch, gen

    def __call__(self):
        ix = torch.randint(0, len(self.ids) - self.seq_len - 1, (self.batch,), generator=self.gen)
        x = torch.stack([self.ids[i:i + self.seq_len] for i in ix])
        y = torch.stack([self.ids[i + 1:i + 1 + self.seq_len] for i in ix])
        return x, y


class QuineBatches:
    """Sequences that ask the model for one row of one of its own matrices.

    Input : <Q> <MAT_m> <hi> <lo> v_0 ... v_{n-2} v_{n-1}
    Target:  -    -      -   v_0  v_1 ... v_{n-1} <END>
    Loss is applied only where the target is a value token or <END>.
    """

    def __init__(self, model: GPT, vocab: Vocab, batch: int, gen: torch.Generator):
        self.model, self.vocab, self.batch, self.gen = model, vocab, batch, gen
        self.mats = model.matrices()
        rows = torch.tensor([m.rows for _, m in self.mats], dtype=torch.float32)
        self.mat_probs = rows / rows.sum()

    def rows_of(self, name: str, mat, rows: torch.Tensor):
        """Build (x, y, mask) for the given row indices of one matrix."""
        q = mat.q()[rows]  # (B, cols) ints, no grad
        B, cols = q.shape
        prefix = torch.tensor([self.vocab.row_prefix(name, int(r)) for r in rows])  # (B, 4)
        vals = q + (self.vocab.VAL0 + self.vocab.k)
        end = torch.full((B, 1), self.vocab.END)
        seq = torch.cat([prefix, vals, end], dim=1)  # (B, 4+cols+1)
        x, y = seq[:, :-1], seq[:, 1:]
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask[:, Vocab.PREFIX_LEN - 1:] = True
        return x, y, mask

    def __call__(self):
        mi = int(torch.multinomial(self.mat_probs, 1, generator=self.gen))
        name, mat = self.mats[mi]
        rows = torch.randint(0, mat.rows, (min(self.batch, mat.rows),), generator=self.gen)
        return self.rows_of(name, mat, rows)

    def scale_batch(self):
        """All per-matrix scales: <Q> <MAT_m> <SCALE> -> S_i <END>."""
        seqs = []
        for name, mat in self.mats:
            seqs.append(self.vocab.scale_prefix(name) + [self.vocab.scale_tok(mat.scale_index()), self.vocab.END])
        seq = torch.tensor(seqs)
        x, y = seq[:, :-1], seq[:, 1:]
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask[:, 2:] = True
        return x, y, mask
