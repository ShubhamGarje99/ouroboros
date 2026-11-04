"""Reproduction with mutation for TableGPT: print yourself with sampling noise, keep the fittest child.

The parent answers every "<Q> <row r> <col c>" by *sampling* among the three value tokens at
temperature tau instead of taking the argmax.  Each sampled printout is loaded into a fresh
child; children are scored by language-model validation loss; the best becomes the next parent.
No gradients anywhere.  With tau -> 0 this is exact replication; with tau > 0 the copier's own
uncertainty about its weights is the mutation operator.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .data import load_text
from .tm import TableGPT
from .train_tm import eval_lm
from .verify_tm import load, self_print


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--data", default="data.noindex/shakespeare.txt")
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--offspring", type=int, default=4)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed)

    parent, ck = load(args.ckpt)
    _, val_text, _ = load_text(args.data)
    val_ids = torch.tensor(parent.encode_text(val_text))
    seq = ck["args"]["seq"]
    parent_loss = eval_lm(parent, val_ids, seq, 32)
    history = [{"generation": 0, "lm_val": parent_loss, "changed": 0}]
    print(f"gen 0: parent lm_val {parent_loss:.4f}")
    for g in range(1, args.generations + 1):
        t0 = time.time()
        pc = parent.codes()
        kids = []
        for i in range(args.offspring):
            printed, s_idx, _ = self_print(parent, tau=args.tau, gen=gen)
            child = TableGPT(parent.cfg)
            child.load_codes(printed, s_idx)
            child.frozen_scale = True
            child.eval()
            diff = int((printed != pc).sum())
            loss = eval_lm(child, val_ids, seq, 32)
            kids.append((loss, diff, child))
            print(f"  gen {g} child {i}: {diff} weights mutated, lm_val {loss:.4f}", flush=True)
        kids.sort(key=lambda t: t[0])
        best_loss, best_diff, best = kids[0]
        history.append({"generation": g, "lm_val": best_loss, "changed": best_diff,
                        "all": [(l, d) for l, d, _ in kids], "seconds": time.time() - t0})
        print(f"gen {g}: best child lm_val {best_loss:.4f} ({best_diff} mutations, {time.time() - t0:.0f}s)", flush=True)
        parent = best
    out = os.path.join(os.path.dirname(args.ckpt), f"evolve_tau{args.tau}.json")
    json.dump(history, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
