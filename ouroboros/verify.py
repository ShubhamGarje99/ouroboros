"""Make the model print itself, rebuild a child from the printout, and compare.

Reports, per generation:
  * exact-match fraction between printed codes and the printer's own codes
  * whether the child is bit-identical to the parent
  * the child's language-model validation loss
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from .data import TextBatches, load_text
from .model import GPT, Config
from .vocab import Vocab


def load(path):
    ck = torch.load(path)
    cfg = Config(**ck["cfg"])
    vocab = Vocab(ck["chars"], cfg.k, ck["mat_names"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, vocab, ck


@torch.no_grad()
def print_matrix(model: GPT, vocab: Vocab, name: str, mat, batch_rows: int = 64):
    """Greedy autoregressive print of every row of one matrix. Returns (codes, n_bad_tokens)."""
    out = torch.zeros(mat.rows, mat.cols, dtype=torch.int64)
    bad = 0
    for r0 in range(0, mat.rows, batch_rows):
        rows = list(range(r0, min(mat.rows, r0 + batch_rows)))
        seq = torch.tensor([vocab.row_prefix(name, r) for r in rows])
        for c in range(mat.cols):
            logits = model(seq)[:, -1]
            # the printout must be a value token: restrict to the value range
            vt = logits[:, vocab.VAL0: vocab.VAL0 + 2 * vocab.k + 1]
            bad += int((logits.argmax(-1) != vt.argmax(-1) + vocab.VAL0).sum())
            nxt = vt.argmax(-1) + vocab.VAL0
            seq = torch.cat([seq, nxt[:, None]], dim=1)
        out[r0:r0 + len(rows)] = seq[:, Vocab.PREFIX_LEN:] - (vocab.VAL0 + vocab.k)
    return out, bad


@torch.no_grad()
def print_scale(model: GPT, vocab: Vocab, name: str) -> int:
    seq = torch.tensor([vocab.scale_prefix(name)])
    logits = model(seq)[0, -1, vocab.SCALE0: vocab.SCALE0 + 16]
    return int(logits.argmax())


@torch.no_grad()
def self_print(model: GPT, vocab: Vocab):
    printed, report = {}, {}
    for name, mat in model.matrices():
        q, bad = print_matrix(model, vocab, name, mat)
        s = print_scale(model, vocab, name)
        q_true, s_true = mat.q(), mat.scale_index()
        exact = float((q == q_true).float().mean())
        printed[name] = (q, s)
        report[name] = {"exact": exact, "scale_ok": s == s_true, "n": q.numel(), "bad_tokens": bad,
                        "rows_exact": float((q == q_true).all(dim=1).float().mean())}
    return printed, report


@torch.no_grad()
def lm_val_loss(model, vocab, val_text, n=20, seq=256, batch=32):
    vb = TextBatches(val_text, vocab, seq, batch, torch.Generator().manual_seed(1))
    tot = 0.0
    for _ in range(n):
        x, y = vb()
        tot += F.cross_entropy(model(x).reshape(-1, model.cfg.vocab_size), y.reshape(-1)).item()
    return tot / n


@torch.no_grad()
def sample_text(model, vocab, prompt="\n", n=300, temp=0.8, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.tensor([vocab.encode_text(prompt)])
    for _ in range(n):
        logits = model(ids[:, -model.cfg.block_size:])[0, -1, : vocab.C] / temp
        nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=g)
        ids = torch.cat([ids, nxt[None]], dim=1)
    return vocab.decode_text(ids[0].tolist())


def identical(a, b, names):
    return all(torch.equal(a[n][0], b[n][0]) and a[n][1] == b[n][1] for n in names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--data", default="data.noindex/shakespeare.txt")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    model, vocab, ck = load(args.ckpt)
    _, val_text, _ = load_text(args.data)
    names = [n for n, _ in model.matrices()]
    out_dir = os.path.dirname(args.ckpt)

    results = {"params": model.n_params(), "cfg": ck["cfg"], "generations": []}
    parent = model
    print(f"parent: {model.n_params():,} params, lm_val {lm_val_loss(model, vocab, val_text):.4f}")
    print("--- parent sample ---\n" + sample_text(model, vocab) + "\n---------------------")
    for g in range(args.generations):
        t0 = time.time()
        printed, rep = self_print(parent, vocab)
        total = sum(r["n"] for r in rep.values())
        exact_all = sum(r["exact"] * r["n"] for r in rep.values()) / total
        child = GPT(parent.cfg)
        child.load_codes(printed)
        child.eval()
        same = identical(printed, parent.codes(), names)
        child_loss = lm_val_loss(child, vocab, val_text)
        rec = {"generation": g + 1, "exact_weights": exact_all, "bit_identical": same,
               "child_lm_val": child_loss, "per_matrix": rep, "seconds": time.time() - t0}
        results["generations"].append(rec)
        print(f"gen {g + 1}: printed {total:,} weights, exact {exact_all:.5f}, "
              f"bit-identical={same}, child lm_val {child_loss:.4f} ({time.time() - t0:.0f}s)")
        for n, r in rep.items():
            print(f"   {n:8s} exact {r['exact']:.4f} rows_exact {r['rows_exact']:.3f} scale_ok {r['scale_ok']} bad {r['bad_tokens']}")
        parent = child
        if same:
            print("   fixed point reached: child == parent")
    json.dump(results, open(os.path.join(out_dir, "verify.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
