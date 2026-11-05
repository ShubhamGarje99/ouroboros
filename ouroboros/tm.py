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

Scales.  A row's weights are its codes times one scale.  With n_scale = 1 there is one
global scale (v1).  With n_scale = 2 the rows of the copy circuit (control, value, coordinate
and scale tokens, positions 0-2, layer-0 attention and the circuit's hidden units) use scale
group 0 and every other row uses group 1, so the copier can run at the large scale it needs
while the language model trains at a small one.  Each scale is printed by its own prompt
<Q> <SCALE_g> and has its own set of scale tokens.
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


def n_circuit_units(d: int) -> int:
    """Hidden units of layer 0 used by the hand-installed copy circuit (see circuit.py):
    3 per printable coordinate + 19 detector/offset units."""
    return 3 * (d - 4) + 19


@dataclass
class TConfig:
    chars: str
    d: int = 64
    n_layer: int = 2
    n_head: int = 4
    block_size: int = 128
    k: int = 1  # ternary
    binary: bool = False  # codes in {-1, +1} only (no zeros)
    strict_causal: bool = False  # positions attend only to earlier positions (position 0 to itself)
    n_scale: int = 1  # scale groups: 1 = one global scale; 2 = circuit rows / free rows


class RMSNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class TableGPT(nn.Module):
    def __init__(self, cfg: TConfig):
        super().__init__()
        self.cfg = cfg
        d, L, G = cfg.d, cfg.n_layer, cfg.n_scale
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
        add("ctl", 2 + G)                 # <Q> <END> <SCALE_0> [<SCALE_1>]
        add("scale", N_SCALE_LEVELS * G)  # scale-index tokens, one set per group
        add("coord", d)                   # one token per coordinate of a row
        add("pos", cfg.block_size)
        for l in range(L):
            for name, n in (("q", d), ("k", d), ("v", d), ("o", d), ("w1", 4 * d), ("g", 4 * d), ("w2T", 4 * d)):
                add(f"L{l}.{name}", n)
        self.R = r
        self.C = C
        self.VAL0 = self.slices["val"][0]
        self.Q, self.END = self.slices["ctl"][0], self.slices["ctl"][0] + 1
        self.SCALE_ADDRS = [self.slices["ctl"][0] + 2 + g for g in range(G)]
        self.SCALE_ADDR = self.SCALE_ADDRS[0]
        self.SCALE0 = self.slices["scale"][0]
        self.COORD0 = self.slices["coord"][0]
        self.POS0 = self.slices["pos"][0]
        # --- scale groups per row ------------------------------------------------------
        group = torch.zeros(self.R, dtype=torch.int64)
        if G == 2:
            a, n = self.slices["chars"]; group[a:a + n] = 1
            group[self.POS0 + 3: self.POS0 + cfg.block_size] = 1
            u = n_circuit_units(d)
            for name in ("w1", "g", "w2T"):
                a, n = self.slices[f"L0.{name}"]; group[a + u: a + n] = 1
            for l in range(1, L):
                for name in ("q", "k", "v", "o", "w1", "g", "w2T"):
                    a, n = self.slices[f"L{l}.{name}"]; group[a:a + n] = 1
        self.register_buffer("row_group", group)
        # --- the table -----------------------------------------------------------
        self.T = nn.Parameter(torch.randn(self.R, d) * 0.02)
        self.register_buffer("scale_idx", torch.zeros(G, dtype=torch.int64))
        self.register_buffer("grid", scale_grid())
        self.frozen = [False] * G
        self.refresh_scale()
        self.norms = nn.ModuleList(RMSNorm() for _ in range(L + 1))

    # --- state-dict compatibility with v1 checkpoints (0-dim scale_idx, no row_group) ----------
    def load_state_dict(self, sd, strict=True):
        sd = dict(sd)
        if sd["scale_idx"].dim() == 0:
            sd["scale_idx"] = sd["scale_idx"].reshape(1)
        sd.setdefault("row_group", self.row_group)
        return super().load_state_dict(sd, strict)

    @property
    def frozen_scale(self) -> bool:
        return all(self.frozen)

    @frozen_scale.setter
    def frozen_scale(self, v: bool):
        self.frozen = [bool(v)] * self.cfg.n_scale

    # --- quantisation --------------------------------------------------------------
    @torch.no_grad()
    def refresh_scale(self):
        for g in range(self.cfg.n_scale):
            if self.frozen[g]:
                continue
            rows = self.row_group == g
            target = self.T[rows].abs().mean() / self.cfg.k
            self.scale_idx[g] = torch.argmin((self.grid - target).abs())

    def scale_values(self) -> torch.Tensor:
        """One scale per group."""
        return self.grid[self.scale_idx]

    def scale_value(self):
        """The scale of group 0 (the only scale when n_scale = 1)."""
        return self.grid[self.scale_idx[0]]

    def row_scale(self) -> torch.Tensor:
        """(R, 1) scale of every row."""
        return self.scale_values()[self.row_group][:, None]

    def _quant(self, w):
        if self.cfg.binary:
            return torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
        return torch.clamp(torch.round(w), -self.cfg.k, self.cfg.k)

    def effective(self):
        if self.training:
            self.refresh_scale()
        s = self.row_scale()
        w = self.T / s
        q = self._quant(w)
        return s * (w + (q - w).detach())

    @torch.no_grad()
    def codes(self) -> torch.Tensor:
        return self._quant(self.T / self.row_scale()).to(torch.int64)

    @torch.no_grad()
    def load_codes(self, q: torch.Tensor, scale_index):
        """Rebuild the table from printed codes and the printed scale index (one per group)."""
        idx = torch.as_tensor(scale_index, dtype=torch.int64).reshape(-1).to(self.scale_idx.device)
        self.scale_idx.copy_(idx)
        self.T.copy_(q.to(self.T.device, torch.float32) * self.row_scale())

    @torch.no_grad()
    def clamp_(self):
        lim = self.row_scale() * (self.cfg.k + 0.49)
        self.T.copy_(torch.minimum(torch.maximum(self.T, -lim), lim))

    def n_params(self):
        return self.R * self.cfg.d

    # --- forward -----------------------------------------------------------------------
    def forward(self, idx: torch.Tensor, head: slice | None = None, last: bool = False) -> torch.Tensor:
        """Logits over all R tokens (or the row slice `head`); with `last`, only for the final position."""
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
            if self.cfg.strict_causal:
                mask = torch.ones(Tn, Tn, dtype=torch.bool, device=idx.device).tril(-1)
                mask[0, 0] = True
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, Tn, d)
            else:
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, Tn, d)
            a = a @ sl("o").T  # attention contribution
            # parallel block: the MLP reads the pre-attention residual, gated by the attention contribution.
            # With W1 = I, G = I this computes coordinate c of a copied row exactly, which is what printing needs.
            hid = F.gelu(h @ sl("w1").T) * (1.0 + a @ sl("g").T)
            x = x + a + hid @ sl("w2T")  # w2 = w2T^T, so hid @ w2^T = hid @ w2T
        x = self.norms[ni](x)
        if last:
            x = x[:, -1]
        if head is not None:
            E = E[head]
        return x @ E.T  # tied head over all R tokens (or the requested slice of them)

    # --- printing helpers -----------------------------------------------------------
    def val_tok(self, q: int) -> int:
        return self.VAL0 + q + self.cfg.k

    def prompt(self, r: int, c: int) -> list[int]:
        """Ask for weight (row r, coordinate c)."""
        return [self.Q, r, self.COORD0 + c]

    def scale_prompt(self, g: int = 0) -> list[int]:
        return [self.Q, self.SCALE_ADDRS[g]]

    def scale_tokens(self, g: int = 0) -> slice:
        """The scale tokens of group g (what the printer decodes among for <Q> <SCALE_g>)."""
        return slice(self.SCALE0 + g * N_SCALE_LEVELS, self.SCALE0 + (g + 1) * N_SCALE_LEVELS)

    def encode_text(self, s: str) -> list[int]:
        return [self.cfg.chars.index(c) for c in s]

    def decode_text(self, ids) -> str:
        return "".join(self.cfg.chars[i] if i < self.C else f"<{i}>" for i in ids)
