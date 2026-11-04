"""Print every row of the table greedily, rebuild a child, compare, repeat for generations."""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from .data import load_text
from .tm import N_SCALE_LEVELS, TableGPT, TConfig
from .train_tm import eval_lm


def load(path):
    ck = torch.load(path)
    cfg = TConfig(**ck["cfg"])
    model = TableGPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


@torch.no_grad()
def self_print(model: TableGPT, batch=4096, tau=0.0, gen=None):
    """Ask for every (row, col) weight. Greedy (tau=0) or sampled. Returns (codes, scale_idx, n_bad)."""
    d = model.cfg.d
    v0, v1 = model.VAL0, model.VAL0 + 2 * model.cfg.k + 1
    rows = torch.arange(model.R).repeat_interleave(d)
    cols = torch.arange(d).repeat(model.R)
    out = torch.zeros(model.R * d, dtype=torch.int64)
    bad = 0
    for i in range(0, len(rows), batch):
        r, c = rows[i:i + batch], cols[i:i + batch]
        x = torch.stack([torch.full_like(r, model.Q), r, model.COORD0 + c], dim=1)
        logits = model(x)[:, -1]
        bad += int(((logits.argmax(-1) < v0) | (logits.argmax(-1) >= v1)).sum())
        vt = logits[:, v0:v1]
        if tau > 0:
            nxt = torch.multinomial(F.softmax(vt / tau, -1), 1, generator=gen)[:, 0]
        else:
            nxt = vt.argmax(-1)
        out[i:i + batch] = nxt - model.cfg.k
    s_logits = model(torch.tensor([model.scale_prompt()]))[0, -1, model.SCALE0: model.SCALE0 + N_SCALE_LEVELS]
    return out.view(model.R, d), int(s_logits.argmax()), bad


@torch.no_grad()
def sample_text(model, prompt="\n", n=400, temp=0.8, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.tensor([model.encode_text(prompt)])
    for _ in range(n):
        logits = model(ids[:, -model.cfg.block_size:])[0, -1, : model.C] / temp
        ids = torch.cat([ids, torch.multinomial(F.softmax(logits, -1), 1, generator=g)[None]], dim=1)
    return model.decode_text(ids[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--data", default="data.noindex/shakespeare.txt")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    model, ck = load(args.ckpt)
    _, val_text, _ = load_text(args.data)
    val_ids = torch.tensor(model.encode_text(val_text))
    seq = ck["args"]["seq"]
    results = {"params": model.n_params(), "rows": model.R, "generations": []}
    parent = model
    print(f"parent: {model.R} rows x {model.cfg.d} = {model.n_params():,} params; lm_val {eval_lm(model, val_ids, seq, 32):.4f}")
    print("--- parent sample ---\n" + sample_text(model) + "\n---------------------")
    for g in range(1, args.generations + 1):
        t0 = time.time()
        printed, s_idx, bad = self_print(parent)
        true = parent.codes()
        exact = float((printed == true).float().mean())
        rows_exact = float((printed == true).all(1).float().mean())
        child = TableGPT(parent.cfg)
        child.load_codes(printed, s_idx)
        child.eval()
        same = bool(torch.equal(printed, true)) and s_idx == int(parent.scale_idx)
        child_loss = eval_lm(child, val_ids, seq, 32)
        rec = {"generation": g, "exact_weights": exact, "rows_exact": rows_exact, "scale_ok": s_idx == int(parent.scale_idx),
               "bit_identical": same, "child_lm_val": child_loss, "bad_tokens": bad, "seconds": time.time() - t0}
        results["generations"].append(rec)
        print(f"gen {g}: exact {exact:.5f} rows_exact {rows_exact:.4f} scale_ok {rec['scale_ok']} bit-identical={same} "
              f"child lm_val {child_loss:.4f} bad_tokens {bad} ({time.time() - t0:.0f}s)", flush=True)
        parent = child
    json.dump(results, open(os.path.join(os.path.dirname(args.ckpt), "verify.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
