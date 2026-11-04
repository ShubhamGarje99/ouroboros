"""A tiny GPT whose every parameter is a small integer times a per-matrix scale.

All learned state lives in `QMatrix` objects (token embeddings, position
embeddings, attention and MLP projections).  Each matrix is stored as a float
"shadow" weight for the optimizer, but the forward pass only ever sees the
quantized value  scale * q  with  q in {-k, ..., k}  (straight-through
estimator), and the scale itself is snapped to a 64-level log grid.  That makes
"the weights" a finite string of small integers, which is what the model is
trained to print.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SCALE_LEVELS = 64
LOG2_SCALE_MIN, LOG2_SCALE_MAX = -12.0, -1.0  # scale grid covers 2^-12 .. 2^-1 in 64 log steps
ROW_BASE = 32  # row addresses are two base-32 digits -> up to 1024 rows


def scale_grid() -> torch.Tensor:
    return 2.0 ** torch.linspace(LOG2_SCALE_MIN, LOG2_SCALE_MAX, N_SCALE_LEVELS)


class QMatrix(nn.Module):
    """A (rows x cols) matrix with integer entries in [-k, k] and one log-grid scale.

    The scale is not learned: in training mode it is recomputed from the mean
    absolute shadow weight (BitNet-style) and snapped to the grid; the snapped
    index is kept in a buffer so that eval mode, checkpoints and printed copies
    all quantize with exactly the same scale.
    """

    sparsity: float = 0.0  # class-level default; GPT sets it from Config

    def __init__(self, rows: int, cols: int, k: int, init_std: float, sparsity: float = 0.0):
        super().__init__()
        self.rows, self.cols, self.k = rows, cols, k
        self.sparsity = sparsity
        self.weight = nn.Parameter(torch.randn(rows, cols) * init_std)
        self.register_buffer("scale_idx", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("grid", scale_grid())
        self.beta = 1.0 if k == 1 else 0.8 * k  # s = beta * mean|w| / k
        self.refresh_scale()

    # --- scale -------------------------------------------------------------
    @torch.no_grad()
    def refresh_scale(self) -> None:
        if self.sparsity > 0:
            # choose the scale so that about `sparsity` of the weights round to zero
            a = self.weight.abs().flatten()
            n = min(a.numel(), 65536)
            if a.numel() > n:
                a = a[torch.randint(0, a.numel(), (n,))]
            target = 2.0 * torch.quantile(a, self.sparsity)
        else:
            target = self.beta * self.weight.abs().mean() / self.k
        self.scale_idx.copy_(torch.argmin((self.grid - target).abs()))

    def scale_index(self) -> int:
        return int(self.scale_idx)

    def scale_value(self) -> torch.Tensor:
        return self.grid[self.scale_idx]

    # --- integer codes -----------------------------------------------------
    def q(self) -> torch.Tensor:
        """Integer codes in [-k, k] (no gradient). Shape (rows, cols)."""
        s = self.scale_value()
        return torch.clamp(torch.round(self.weight.detach() / s), -self.k, self.k).to(torch.int64)

    def effective(self) -> torch.Tensor:
        """scale * q with a straight-through gradient to the shadow weight."""
        if self.training:
            self.refresh_scale()
        s = self.scale_value()
        w = self.weight / s
        q = torch.clamp(torch.round(w), -self.k, self.k)
        return s * (w + (q - w).detach())

    @torch.no_grad()
    def clamp_(self) -> None:
        """Keep the shadow weight inside the representable range [-k*s, k*s]."""
        lim = float(self.scale_value()) * (self.k + 0.49)
        self.weight.clamp_(-lim, lim)

    @torch.no_grad()
    def load_codes(self, q: torch.Tensor, scale_index: int) -> None:
        """Overwrite this matrix from printed integers + scale index (exact)."""
        self.scale_idx.fill_(scale_index)
        self.weight.copy_(q.to(torch.float32) * self.grid[scale_index])


class RMSNorm(nn.Module):
    """Parameter-free RMSNorm so that there is nothing extra to print."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class Block(nn.Module):
    def __init__(self, d: int, n_head: int, k: int, sparsity: float = 0.0):
        super().__init__()
        self.d, self.n_head = d, n_head
        std = 0.02
        self.q = QMatrix(d, d, k, std, sparsity)
        self.k_ = QMatrix(d, d, k, std, sparsity)
        self.v = QMatrix(d, d, k, std, sparsity)
        self.o = QMatrix(d, d, k, std, sparsity)
        self.w1 = QMatrix(4 * d, d, k, std, sparsity)
        self.w2 = QMatrix(d, 4 * d, k, std, sparsity)
        self.n1, self.n2 = RMSNorm(), RMSNorm()

    def matrices(self, prefix: str) -> list[tuple[str, QMatrix]]:
        return [(f"{prefix}.{n}", m) for n, m in
                (("q", self.q), ("k", self.k_), ("v", self.v), ("o", self.o), ("w1", self.w1), ("w2", self.w2))]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        h = self.n1(x)
        q = (h @ self.q.effective().T).view(B, T, self.n_head, -1).transpose(1, 2)
        k = (h @ self.k_.effective().T).view(B, T, self.n_head, -1).transpose(1, 2)
        v = (h @ self.v.effective().T).view(B, T, self.n_head, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, d)
        x = x + a @ self.o.effective().T
        h = self.n2(x)
        x = x + F.gelu(h @ self.w1.effective().T) @ self.w2.effective().T
        return x


@dataclass
class Config:
    vocab_size: int
    block_size: int = 288
    d: int = 64
    n_layer: int = 2
    n_head: int = 4
    k: int = 3  # integer codes in [-k, k]  -> 2k+1 levels
    tie_head: bool = True
    sparsity: float = 0.0  # target fraction of zero codes (0 = BitNet-style mean|w| scale)


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        sp = cfg.sparsity
        self.tok = QMatrix(cfg.vocab_size, cfg.d, cfg.k, 0.02, sp)
        self.pos = QMatrix(cfg.block_size, cfg.d, cfg.k, 0.02, sp)
        self.blocks = nn.ModuleList(Block(cfg.d, cfg.n_head, cfg.k, sp) for _ in range(cfg.n_layer))
        self.head = None if cfg.tie_head else QMatrix(cfg.vocab_size, cfg.d, cfg.k, 0.02, sp)
        self.norm = RMSNorm()

    # --- the registry of everything that has to be printed -----------------
    def matrices(self) -> list[tuple[str, QMatrix]]:
        out = [("tok", self.tok), ("pos", self.pos)]
        for i, b in enumerate(self.blocks):
            out += b.matrices(f"L{i}")
        if self.head is not None:
            out.append(("head", self.head))
        return out

    def n_params(self) -> int:
        return sum(m.rows * m.cols for _, m in self.matrices())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        E = self.tok.effective()
        x = E[idx] + self.pos.effective()[:T]
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        W = E if self.head is None else self.head.effective()
        return x @ W.T

    # --- codes in / codes out ------------------------------------------------
    @torch.no_grad()
    def codes(self) -> dict[str, tuple[torch.Tensor, int]]:
        return {name: (m.q(), m.scale_index()) for name, m in self.matrices()}

    @torch.no_grad()
    def load_codes(self, codes: dict[str, tuple[torch.Tensor, int]]) -> None:
        for name, m in self.matrices():
            q, s = codes[name]
            m.load_codes(q, s)

    @torch.no_grad()
    def clamp_(self) -> None:
        for _, m in self.matrices():
            m.clamp_()

    @torch.no_grad()
    def code_stats(self) -> dict[str, float]:
        qs = torch.cat([q.flatten() for q, _ in self.codes().values()])
        return {"n": int(qs.numel()), "frac_zero": float((qs == 0).float().mean()),
                "frac_sat": float((qs.abs() == self.cfg.k).float().mean())}
