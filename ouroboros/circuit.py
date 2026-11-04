"""Hand-installed copy circuit for TableGPT (layer 0), as a constructive proof.

Prompt: <Q> <row r> <col c>.  Reserved coordinates of the d-dim stream:
  M  (=d-1): attention marker.  pos1 = +e_M, pos2 = -e_M.  Every head's query reads -x[M]
             and every head's key reads +x[M], so position 2 attends to position 1
             (score ~ +3) and not to itself (score ~ -8).
  K  (=d-2): constant marker. pos2 also carries +e_K; two gate-free hidden units read it
             and feed the "0" logit coordinate.
  A, Z (=d-3, d-4): readout coordinates.  Value token embeddings are +e_A (+1), -e_A (-1), +e_Z (0).
Heads copy the attended row (W_v = W_o = I); the MLP gate is the identity on the attention
output, so hidden unit c = GELU(x2[c]) * (1 + a[c]) is zero unless c is the asked coordinate,
where it is one of three levels depending on T[r][c].  W2 sends it to coordinate A with an
offset so the levels are about -11, 0, +11, and sends a constant ~9 to coordinate Z.
Layer 1 is zeroed (identity map).  Coordinates M and K cannot be printed by the circuit.
"""
from __future__ import annotations

import torch

from .tm import TableGPT


@torch.no_grad()
def install_copy_circuit(model: TableGPT, seed: int = 0, scale: float = 1.5):
    """Weights of order one are needed for the attention scores and the gate to be crisp,
    so the global scale is pinned near `scale` and frozen."""
    d, H = model.cfg.d, model.cfg.n_head
    hd = d // H
    M, K, A, Z = d - 1, d - 2, d - 3, d - 4
    s = model.slices
    codes = model.codes().clone()
    codes.zero_()
    g = torch.Generator().manual_seed(seed)

    def rows(name):
        a, n = s[name]
        return slice(a, a + n)

    # text characters, scale tokens: random sparse ternary, reserved coords zero
    for name in ("chars", "scale"):
        r = rows(name)
        blk = torch.randint(-1, 2, (r.stop - r.start, d), generator=g)
        blk[torch.rand(blk.shape, generator=g) < 0.6] = 0
        blk[:, [M, K, A, Z]] = 0
        codes[r] = blk
    # control tokens: <Q> = 0 (contributes nothing), <END>, <SCALE> random-ish
    q0 = s["ctl"][0]
    codes[q0 + 1, 0] = 1
    codes[q0 + 2, 1] = 1
    # value tokens
    v0 = model.VAL0
    codes[v0 + 0, A] = -1   # value -1
    codes[v0 + 1, Z] = 1    # value 0
    codes[v0 + 2, A] = 1    # value +1
    # coordinate tokens: one-hot
    c0 = model.COORD0
    for c in range(d):
        codes[c0 + c, c] = 1
    # positions: 0 -> zero, 1 -> +e_M, 2 -> -e_M + e_K, rest random sparse (reserved coords zero)
    p0 = model.POS0
    codes[p0 + 1, M] = 1
    codes[p0 + 2, M] = -1
    codes[p0 + 2, K] = 1
    pr = torch.randint(-1, 2, (model.cfg.block_size - 3, d), generator=g)
    pr[torch.rand(pr.shape, generator=g) < 0.7] = 0
    pr[:, [M, K, A, Z]] = 0
    codes[p0 + 3: p0 + model.cfg.block_size] = pr
    # layer 0 attention
    Wq, Wk, Wv, Wo = (codes[rows(f"L0.{n}")] for n in ("q", "k", "v", "o"))
    for h in range(H):
        Wq[h * hd, M] = -1
        Wk[h * hd, M] = 1
    Wv.copy_(torch.eye(d, dtype=codes.dtype))
    Wo.copy_(torch.eye(d, dtype=codes.dtype))
    # layer 0 MLP: W1 (4d x d), G (4d x d), W2T (4d x d)
    W1, G, W2T = (codes[rows(f"L0.{n}")] for n in ("w1", "g", "w2T"))
    for c in range(d):
        if c in (M, K):
            continue
        W1[c, c] = 1
        G[c, c] = 1
        W2T[c, A] = 1
    # constant units: read coordinate K (gate-free)
    W1[d, K] = 1
    W2T[d, A] = -1          # offset the readout so the middle level sits near 0
    W1[d + 1, K] = 1
    W2T[d + 1, Z] = 1       # "0" logit constant
    W1[d + 2, K] = 1
    W2T[d + 2, Z] = 1
    # layer 1: identity map (all zero)
    idx = int(torch.argmin((model.grid - scale).abs()))
    model.load_codes(codes, idx)
    model.frozen_scale = True
    return model


def circuit_rows(model: TableGPT) -> torch.Tensor:
    """Indices of the rows the hand-installed circuit depends on (for damped updates)."""
    s = model.slices
    idx = [model.Q, model.POS0 + 1, model.POS0 + 2]
    for name in ("val", "coord", "L0.q", "L0.k", "L0.v", "L0.o", "L0.w1", "L0.g", "L0.w2T"):
        a, n = s[name]
        idx += list(range(a, a + n))
    return torch.tensor(sorted(set(idx)))
