"""Reproduction with mutation: the parent prints itself *stochastically*.

Instead of taking the argmax value token at every position, we sample from the
model's own distribution over value tokens at temperature `tau`.  Each sampled
printout is loaded into a fresh child.  Children are scored on language-model
validation loss, the best child becomes the next parent, and the process
repeats.  No gradients are used anywhere: the only source of change is the
copier's own uncertainty about its weights.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from .data import load_text
from .model import GPT
from .verify import identical, lm_val_loss, load, print_scale
from .vocab import Vocab


@torch.no_grad()
def print_matrix_sampled(model, vocab, name, mat, tau, gen, batch_rows=64):
    out = torch.zeros(mat.rows, mat.cols, dtype=torch.int64)
    for r0 in range(0, mat.rows, batch_rows):
        rows = list(range(r0, min(mat.rows, r0 + batch_rows)))
        seq = torch.tensor([vocab.row_prefix(name, r) for r in rows])
        for _ in range(mat.cols):
            vt = model(seq)[:, -1, vocab.VAL0: vocab.VAL0 + 2 * vocab.k + 1]
            if tau > 0:
                nxt = torch.multinomial(F.softmax(vt / tau, -1), 1, generator=gen)[:, 0]
            else:
                nxt = vt.argmax(-1)
            seq = torch.cat([seq, (nxt + vocab.VAL0)[:, None]], dim=1)
        out[r0:r0 + len(rows)] = seq[:, Vocab.PREFIX_LEN:] - (vocab.VAL0 + vocab.k)
    return out


@torch.no_grad()
def reproduce(parent, vocab, tau, gen):
    codes = {}
    for name, mat in parent.matrices():
        codes[name] = (print_matrix_sampled(parent, vocab, name, mat, tau, gen), print_scale(parent, vocab, name))
    child = GPT(parent.cfg)
    child.load_codes(codes)
    child.eval()
    return child, codes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--data", default="data.noindex/shakespeare.txt")
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--offspring", type=int, default=3)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed)

    parent, vocab, _ = load(args.ckpt)
    _, val_text, _ = load_text(args.data)
    names = [n for n, _ in parent.matrices()]
    out_dir = os.path.dirname(args.ckpt)
    parent_loss = lm_val_loss(parent, vocab, val_text)
    history = [{"generation": 0, "lm_val": parent_loss}]
    print(f"gen 0 parent lm_val {parent_loss:.4f}")
    for g in range(1, args.generations + 1):
        t0 = time.time()
        kids = []
        for i in range(args.offspring):
            child, codes = reproduce(parent, vocab, args.tau, gen)
            pc = parent.codes()
            diff = sum(int((codes[n][0] != pc[n][0]).sum()) for n in names)
            loss = lm_val_loss(child, vocab, val_text)
            kids.append((loss, diff, child))
            print(f"  gen {g} child {i}: {diff} weights changed, lm_val {loss:.4f}", flush=True)
        kids.sort(key=lambda t: t[0])
        best_loss, best_diff, best = kids[0]
        history.append({"generation": g, "lm_val": best_loss, "changed": best_diff,
                        "all": [(l, d) for l, d, _ in kids], "seconds": time.time() - t0})
        print(f"gen {g}: best child lm_val {best_loss:.4f} ({best_diff} weights mutated, {time.time() - t0:.0f}s)", flush=True)
        parent = best
    json.dump(history, open(os.path.join(out_dir, f"evolve_tau{args.tau}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
