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
    """Install the copy circuit in layer 0 (layer 1 is zero = identity).

    Prompt <Q> <row r> <col c>.  No marker is placed on the row position, so the attended
    vector is exactly the normalised row.  Reserved coordinates (all printable):
      Q0 = d-1 : <Q> = e_Q0.                          keys read -x[Q0] - x[J] - x[J2]
      J, J2 = d-2, d-3 : pos2 = e_J + e_J2.           queries read x[J]
      A  = d-4 : readout (+1 -> +e_A, -1 -> -e_A).    Z = d-5 : readout for 0 (+e_Z)
    Positions 0 and 1 have zero embeddings, so position 2 attends to position 1 because the
    other keys are strongly negative.  Hidden units (two copies each for gain): identity units
    for every coordinate but J, J2; difference units x[J]-x[J2] and x[J2]-x[J]; two offset
    units reading x[J] into -A; one unit reading x[J2] into +Z.  Scale pinned near `scale`."""
    d, H = model.cfg.d, model.cfg.n_head
    hd = d // H
    Q0, J, J2, A, Z = d - 1, d - 2, d - 3, d - 4, d - 5
    reserved = [Q0, J, J2, A, Z]
    s = model.slices
    codes = model.codes().clone()
    codes.zero_()
    g = torch.Generator().manual_seed(seed)

    def rows(name):
        a, n = s[name]
        return slice(a, a + n)

    for name in ("chars", "scale"):
        r = rows(name)
        blk = torch.randint(-1, 2, (r.stop - r.start, d), generator=g)
        blk[torch.rand(blk.shape, generator=g) < 0.6] = 0
        blk[:, reserved] = 0
        codes[r] = blk
    q0 = s["ctl"][0]
    codes[q0 + 0, Q0] = 1         # <Q>
    codes[q0 + 1, 0] = 1          # <END>
    codes[q0 + 2, 1] = 1          # <SCALE> = e_1
    codes[rows("scale"), 1] = 0   # only the correct scale token will carry e_1
    v0 = model.VAL0
    codes[v0 + 0, A] = -1
    codes[v0 + 1, Z] = 1
    codes[v0 + 2, A] = 1
    c0 = model.COORD0
    for c in range(d):
        codes[c0 + c, c] = 1
    p0 = model.POS0
    codes[p0 + 2, J] = 1
    codes[p0 + 2, J2] = 1
    pr = torch.randint(-1, 2, (model.cfg.block_size - 3, d), generator=g)
    pr[torch.rand(pr.shape, generator=g) < 0.7] = 0
    pr[:, reserved] = 0
    codes[p0 + 3: p0 + model.cfg.block_size] = pr
    Wq, Wk, Wv, Wo = (codes[rows(f"L0.{n}")] for n in ("q", "k", "v", "o"))
    assert model.cfg.strict_causal, "the construction needs strictly causal attention (no self-attention)"
    for h in range(H):
        # position 2 chooses between <Q> (key -8 x 4) and the row (key ~0): the row wins for every row
        for i in range(4):
            Wq[h * hd + i, J] = 1
            Wk[h * hd + i, Q0] = -1
    Wv.copy_(torch.eye(d, dtype=codes.dtype))
    Wo.copy_(torch.eye(d, dtype=codes.dtype))
    W1, G, W2T = (codes[rows(f"L0.{n}")] for n in ("w1", "g", "w2T"))
    # Unit counts are chosen so that, after RMSNorm, the offset equals the "value 0" level exactly:
    # ordinary c: 3 identity units x 4.62 = offsets (2 x[J] + x[J2]) = 3 x 4.62;
    # c = J:  5 detector units x 3.58 = 2 x 7.16 + 3.58;   c = J2: 4 x 3.58 = 2 x 3.58 + 7.16.
    u = 0
    for c in range(d):
        if c in (J, J2):
            continue
        for _ in range(3):
            W1[u, c] = 1; G[u, c] = 1; W2T[u, A] = 1; u += 1
    for _ in range(5):
        W1[u, J] = 1; W1[u, J2] = -1; G[u, J] = 1; W2T[u, A] = 1; u += 1
    for _ in range(4):
        W1[u, J2] = 1; W1[u, J] = -1; G[u, J2] = 1; W2T[u, A] = 1; u += 1
    for _ in range(2):
        W1[u, J] = 1; W2T[u, A] = -1; u += 1    # offset
    W1[u, J2] = 1; W2T[u, A] = -1; u += 1       # offset
    W1[u, J] = 1; W1[u, J2] = 1; W2T[u, Z] = 1; u += 1   # "0" logit >= 9.24 > any residual leak (<= 8)
    assert u <= 4 * d
    idx = int(torch.argmin((model.grid - scale).abs()))
    codes[model.SCALE0 + idx] = 0
    codes[model.SCALE0 + idx, 1] = 1
    model.load_codes(codes, idx)
    model.frozen_scale = True
    return model


def circuit_rows(model: TableGPT) -> torch.Tensor:
    """Indices of the rows the hand-installed circuit depends on (for damped updates)."""
    s = model.slices
    idx = [model.Q, model.POS0 + 1, model.POS0 + 2]
    for name in ("ctl", "scale", "val", "coord", "L0.q", "L0.k", "L0.v", "L0.o", "L0.w1", "L0.g", "L0.w2T"):
        a, n = s[name]
        idx += list(range(a, a + n))
    return torch.tensor(sorted(set(idx)))
