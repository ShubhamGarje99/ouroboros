"""Hand-installed copy circuit for TableGPT (layer 0), as a constructive proof.

Prompt <Q> <row r> <col c>.  Seven coordinates of the d-dimensional stream are reserved
(every one of them printable):
  Q0     : <Q> = e_Q0.  Every head's keys read -x[Q0] in four dimensions, so at the answer
           position the row (key ~ 0) always beats <Q> (key ~ -sqrt(d) * 4).
  J, J2  : position 2 = e_J + e_J2; every head's queries read x[J].  Detector units
           x[J] - x[J2] and x[J2] - x[J] print J and J2 themselves.
  Y, Y2  : text marker, +1 in every position row >= 3 and absent from the prompt positions.
           Every circuit unit subtracts x[Y] + x[Y2], so the circuit is silent on ordinary text.
  A, Z   : readout.  value +1 -> +e_A, -1 -> -e_A, 0 -> +e_Z.
Heads copy the attended row (W_v = W_o = I); the MLP gate is the identity on the attention
output, so three identity units per coordinate compute GELU(x[c] - x[Y] - x[Y2]) * (1 + a[c]),
which is zero unless c is the asked coordinate; unit counts for the detectors and the offset
units are chosen so that, after RMSNorm, the offset equals the "value 0" level exactly.

The scale is printed by construction too: <SCALE> = e_Z and the one correct scale token is
e_Z, every other scale row is zero, so the answer position keeps x[Z] = 1 > 0 = every wrong
scale logit.

Everything outside the circuit is free for the language model: the character and position
rows, the unused hidden units of layer 0, and every later layer.  None of it can disturb a
printed answer, because the value and scale tokens are read only from the reserved
coordinates A and Z, and no free row may write there (W_o rows A, Z and the A, Z columns of
every W_2 are held at zero by `circuit_mask`).  Later layers are also initialised silent on
the prompt (their MLP units subtract the markers J, J2 that only position 2 carries).
"""
from __future__ import annotations

import torch

from .tm import TableGPT


def n_circuit_units(d: int) -> int:
    """Hidden units of layer 0 used by the circuit: 3 per printable coordinate + 19 detector/offset units."""
    return 3 * (d - 4) + 19


@torch.no_grad()
def install_copy_circuit(model: TableGPT, seed: int = 0, scale: float = 1.5, free_density: float = 0.5):
    """Install the copy circuit in layer 0 of a fresh table; random ternary everywhere that is free.

    `free_density` is the fraction of non-zero codes in the free rows of layer >= 1 and the free
    hidden units of layer 0.  Needs strictly causal attention."""
    assert model.cfg.strict_causal, "the construction needs strictly causal attention (no self-attention)"
    assert not model.cfg.binary and model.cfg.k == 1, "the construction is for ternary codes"
    d, H = model.cfg.d, model.cfg.n_head
    hd = d // H
    assert hd >= 4, "each head needs at least 4 dimensions for the marker keys"
    Q0, J, J2, Y, Y2, A, Z = d - 1, d - 2, d - 3, d - 4, d - 5, d - 6, d - 7
    reserved = [Q0, J, J2, Y, Y2, A, Z]
    s = model.slices
    codes = model.codes().clone()
    codes.zero_()
    g = torch.Generator().manual_seed(seed)

    def rows(name):
        a, n = s[name]
        return slice(a, a + n)

    def random_block(n, density):
        blk = torch.randint(-1, 2, (n, d), generator=g)
        blk[torch.rand(blk.shape, generator=g) >= density] = 0
        blk[:, reserved] = 0
        return blk

    codes[rows("chars")] = random_block(s["chars"][1], 0.4)
    q0 = s["ctl"][0]
    codes[q0 + 0, Q0] = 1         # <Q>
    codes[q0 + 1, 0] = 1          # <END>
    codes[q0 + 2, Z] = 1          # <SCALE> = e_Z, read only from a reserved coordinate
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
    pr = random_block(model.cfg.block_size - 3, 0.3)
    pr[:, [Y, Y2]] = 1            # text marker on every ordinary position
    codes[p0 + 3: p0 + model.cfg.block_size] = pr
    Wq, Wk, Wv, Wo = (codes[rows(f"L0.{n}")] for n in ("q", "k", "v", "o"))
    for h in range(H):
        for i in range(4):
            Wq[h * hd + i, J] = 1
            Wk[h * hd + i, Q0] = -1
    Wv.copy_(torch.eye(d, dtype=codes.dtype))
    Wo.copy_(torch.eye(d, dtype=codes.dtype))
    W1, G, W2T = (codes[rows(f"L0.{n}")] for n in ("w1", "g", "w2T"))
    u = 0

    def unit(read: dict, gate: int | None, out: int, sign: int = 1, silence: bool = False):
        nonlocal u
        for cc, v in read.items():
            W1[u, cc] = v
        if silence:  # identity units read x[c], which is nonzero on text: subtract the text markers
            W1[u, Y] = -1
            W1[u, Y2] = -1
        if gate is not None:
            G[u, gate] = 1
        W2T[u, out] = sign
        u += 1

    for c in range(d):
        if c in (J, J2, Y, Y2):
            continue
        for _ in range(3):
            unit({c: 1}, c, A, silence=True)
    for _ in range(5):
        unit({J: 1, J2: -1}, J, A)
    for _ in range(4):
        unit({J2: 1, J: -1}, J2, A)
    for _ in range(3):
        unit({Y: 1, Y2: -1}, Y, A)
        unit({Y2: 1, Y: -1}, Y2, A)
    for _ in range(2):
        unit({J: 1}, None, A, -1)          # offset
    unit({J2: 1}, None, A, -1)             # offset
    unit({J: 1, J2: 1}, None, Z)           # "0" logit: 2*sqrt(d/3) after the norm, above any residual leak
    assert u == n_circuit_units(d) <= 4 * d
    # the remaining hidden units of layer 0 are free: random, silent on the prompt, never writing A or Z
    nfree = 4 * d - u
    for name, blk in (("w1", W1), ("g", G), ("w2T", W2T)):
        b = random_block(nfree, free_density)
        if name == "w1":
            b[:, J] = -1
            b[:, J2] = -1
        blk[u:] = b
    # later layers: random ternary so the language model has a network to train from step 0, but silent on
    # the self-print prompt: their MLP units subtract the query markers (present only at position 2), and they
    # never write the readout coordinates (W_o rows A, Z and every reserved column are zero).
    for l in range(1, model.cfg.n_layer):
        for name in ("q", "k", "v", "o", "w1", "g", "w2T"):
            r = rows(f"L{l}.{name}")
            blk = random_block(r.stop - r.start, free_density)
            if name == "w1":
                blk[:, J] = -1
                blk[:, J2] = -1
            if name == "o":
                blk[[A, Z]] = 0
            codes[r] = blk
    idx = int(torch.argmin((model.grid - scale).abs()))
    codes[rows("scale")] = 0
    codes[model.SCALE0 + idx, Z] = 1
    model.load_codes(codes, idx)
    model.frozen_scale = True
    return model


def circuit_mask(model: TableGPT) -> torch.Tensor:
    """(R, d) boolean mask of entries that must not change during training: every entry of the
    circuit's own rows, and the reserved coordinates of every row."""
    d = model.cfg.d
    m = torch.zeros(model.R, d, dtype=torch.bool)
    for name in ("ctl", "scale", "val", "coord", "L0.q", "L0.k", "L0.v", "L0.o"):
        a, n = model.slices[name]
        m[a:a + n] = True
    u = n_circuit_units(d)
    for name in ("L0.w1", "L0.g", "L0.w2T"):
        a, _ = model.slices[name]
        m[a:a + u] = True
    m[model.POS0: model.POS0 + 3] = True
    m[:, d - 7:] = True
    for l in range(1, model.cfg.n_layer):
        a, n = model.slices[f"L{l}.o"]
        m[[a + d - 6, a + d - 7]] = True  # W_o rows for the readout coordinates A, Z stay zero
    return m


def circuit_rows(model: TableGPT) -> torch.Tensor:
    """Indices of the rows the hand-installed circuit depends on (for damped updates)."""
    s = model.slices
    idx = [model.Q, model.POS0 + 1, model.POS0 + 2]
    for name in ("ctl", "scale", "val", "coord", "L0.q", "L0.k", "L0.v", "L0.o"):
        a, n = s[name]
        idx += list(range(a, a + n))
    u = n_circuit_units(model.cfg.d)
    for name in ("L0.w1", "L0.g", "L0.w2T"):
        a, _ = s[name]
        idx += list(range(a, a + u))
    return torch.tensor(sorted(set(idx)))
