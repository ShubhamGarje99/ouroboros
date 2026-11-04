"""TableGPT: a transformer whose weights are its vocabulary.

All parameters live in ONE ternary table T of shape (R, d).  Every row of T is
simultaneously (a) the embedding of token number r and (b) one row of some
weight matrix: the first rows are the embeddings of text characters, value
tokens and control tokens; then the position embeddings; then, for each layer,
the rows of W_q, W_k, W_v, W_o, W_1 and W_2^T.  Because every weight row is a
token, the prompt  <Q> <row r> <col c>  puts row r itself into the residual
stream, and answering with T[r][c] is a copy operation the model can learn once
and apply to any of its weights, instead of a memorisation problem that cannot
fit (the bits needed to describe the weights equal the bits the weights can
hold).  The MLP is gated by the attention output so that "coordinate c of the
copied row" is a near-linear circuit.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SCALE_LEVELS = 64
LOG2_SCALE_MIN, LOG2_SCALE_MAX = -12.0, 1.0


def scale_grid() -> torch.Tensor:
    return 2.0 ** torch.linspace(LOG2_SCALE_MIN, LOG2_SCALE_MAX, N_SCALE_LEVELS)


@dataclass
class TConfig:
    chars: str
    d: int = 64
    n_layer: int = 2
    n_head: int = 4
    block_size: int = 128
    k: int = 1  # ternary


class RMSNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class TableGPT(nn.Module):
    def __init__(self, cfg: TConfig):
        super().__init__()
        self.cfg = cfg
        d, L = cfg.d, cfg.n_layer
        C = len(cfg.chars)
        # --- row layout (every row is a token) ---------------------------------
        self.slices: dict[str, tuple[int, int]] = {}
        r = 0

        def add(name, n):
            nonlocal r
            self.slices[name] = (r, n)
            r += n

        add("chars", C)
        add("val", 2 * cfg.k + 1)
        add("ctl", 3)                 # <Q> <END> <SCALE>
        add("scale", N_SCALE_LEVELS)  # scale-index tokens
        add("coord", d)               # one token per coordinate of a row
        add("pos", cfg.block_size)
        for l in range(L):
            for name, n in (("q", d), ("k", d), ("v", d), ("o", d), ("w1", 4 * d), ("g", 4 * d), ("w2T", 4 * d)):
                add(f"L{l}.{name}", n)
        self.R = r
        self.C = C
        self.VAL0 = self.slices["val"][0]
        self.Q, self.END, self.SCALE_ADDR = (self.slices["ctl"][0] + i for i in range(3))
        self.SCALE0 = self.slices["scale"][0]
        self.COORD0 = self.slices["coord"][0]
        self.POS0 = self.slices["pos"][0]
        # --- the table -----------------------------------------------------------
        self.T = nn.Parameter(torch.randn(self.R, d) * 0.02)
        self.register_buffer("scale_idx", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("grid", scale_grid())
        self.frozen_scale = False
        self.refresh_scale()
        self.norms = nn.ModuleList(RMSNorm() for _ in range(L + 1))

    # --- quantisation --------------------------------------------------------------
    @torch.no_grad()
    def refresh_scale(self):
        if self.frozen_scale:
            return
        target = self.T.abs().mean() / self.cfg.k
        self.scale_idx.copy_(torch.argmin((self.grid - target).abs()))

    def scale_value(self):
        return self.grid[self.scale_idx]

    def effective(self):
        if self.training:
            self.refresh_scale()
        s = self.scale_value()
        w = self.T / s
        q = torch.clamp(torch.round(w), -self.cfg.k, self.cfg.k)
        return s * (w + (q - w).detach())

    @torch.no_grad()
    def codes(self) -> torch.Tensor:
        s = self.scale_value()
        return torch.clamp(torch.round(self.T / s), -self.cfg.k, self.cfg.k).to(torch.int64)

    @torch.no_grad()
    def load_codes(self, q: torch.Tensor, scale_index: int):
        self.scale_idx.fill_(scale_index)
        self.T.copy_(q.to(torch.float32) * self.grid[scale_index])

    @torch.no_grad()
    def clamp_(self):
        lim = float(self.scale_value()) * (self.cfg.k + 0.49)
        self.T.clamp_(-lim, lim)

    def n_params(self):
        return self.R * self.cfg.d

    # --- forward -----------------------------------------------------------------------
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, Tn = idx.shape
        assert Tn <= self.cfg.block_size
        E = self.effective()
        d, H = self.cfg.d, self.cfg.n_head
        x = E[idx] + E[self.POS0: self.POS0 + Tn]
        ni = 0
        for l in range(self.cfg.n_layer):
            sl = lambda n: E[self.slices[f"L{l}.{n}"][0]: self.slices[f"L{l}.{n}"][0] + self.slices[f"L{l}.{n}"][1]]
            h = self.norms[ni](x); ni += 1
            q = (h @ sl("q").T).view(B, Tn, H, -1).transpose(1, 2)
            k = (h @ sl("k").T).view(B, Tn, H, -1).transpose(1, 2)
            v = (h @ sl("v").T).view(B, Tn, H, -1).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, Tn, d)
            a = a @ sl("o").T  # attention contribution
            # parallel block: the MLP reads the pre-attention residual, gated by the attention contribution.
            # With W1 = I, G = I this computes coordinate c of a copied row exactly, which is what printing needs.
            hid = F.gelu(h @ sl("w1").T) * (1.0 + a @ sl("g").T)
            x = x + a + hid @ sl("w2T")  # w2 = w2T^T, so hid @ w2^T = hid @ w2T
        x = self.norms[ni](x)
        return x @ E.T  # tied head over all R tokens

    # --- printing helpers -----------------------------------------------------------
    def val_tok(self, q: int) -> int:
        return self.VAL0 + q + self.cfg.k

    def prompt(self, r: int, c: int) -> list[int]:
        """Ask for weight (row r, coordinate c)."""
        return [self.Q, r, self.COORD0 + c]

    def scale_prompt(self) -> list[int]:
        return [self.Q, self.SCALE_ADDR]

    def encode_text(self, s: str) -> list[int]:
        return [self.cfg.chars.index(c) for c in s]

    def decode_text(self, ids) -> str:
        return "".join(self.cfg.chars[i] if i < self.C else f"<{i}>" for i in ids)
