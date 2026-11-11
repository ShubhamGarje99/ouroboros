"""Print every row of the table greedily, rebuild a child, compare, repeat for generations."""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from .data import load_text
from .tm import TableGPT, TConfig
from .train_tm import eval_lm


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu")
    cfg = TConfig(**ck["cfg"])
    model = TableGPT(cfg)
    model.load_state_dict(ck["model"])
    model.frozen_scale = True  # a saved model never re-estimates its scale
    model.to(device)
    model.eval()
    return model, ck


@torch.no_grad()
def self_print(model: TableGPT, batch=4096, tau=0.0, gen=None, margins: dict | None = None):
    """Ask for every (row, col) weight. Greedy (tau=0) or sampled. Returns (codes, scale_idx, n_bad).

    `n_bad` counts prompts whose unrestricted argmax is not a value token (the printer decodes among
    the value tokens only).  If `margins` is a dict, it receives the smallest *relative* logit gap
    (true value token minus the best wrong value token, divided by the largest value-token logit
    magnitude, so the final RMSNorm's scaling drops out) over all prompts, and the same for the scale."""
    model.eval()
    dev = model.T.device
    d = model.cfg.d
    v0, v1 = model.VAL0, model.VAL0 + 2 * model.cfg.k + 1
    rows = torch.arange(model.R).repeat_interleave(d)
    cols = torch.arange(d).repeat(model.R)
    out = torch.zeros(model.R * d, dtype=torch.int64)
    true = model.codes().view(-1).cpu() if margins is not None else None
    bad, min_margin = 0, float("inf")
    for i in range(0, len(rows), batch):
        r, c = rows[i:i + batch].to(dev), cols[i:i + batch].to(dev)
        x = torch.stack([torch.full_like(r, model.Q), r, model.COORD0 + c], dim=1)
        logits = model(x, last=True)
        am = logits.argmax(-1)
        bad += int(((am < v0) | (am >= v1)).sum())
        vt = logits[:, v0:v1]
        if tau > 0:
            nxt = torch.multinomial(F.softmax(vt / tau, -1), 1, generator=gen)[:, 0]
        else:
            nxt = vt.argmax(-1)
        out[i:i + batch] = (nxt - model.cfg.k).cpu()
        if margins is not None:
            t = (true[i:i + batch] + model.cfg.k).to(dev)
            lt = vt.gather(1, t[:, None])[:, 0]
            wrong = vt.scatter(1, t[:, None], float("-inf")).max(-1).values
            min_margin = min(min_margin, float(((lt - wrong) / vt.abs().max(-1).values.clamp_min(1e-30)).min()))
    s_idx, s_margin = [], float("inf")
    for g in range(model.cfg.n_scale):
        s_logits = model(torch.tensor([model.scale_prompt(g)], device=dev), head=model.scale_tokens(g), last=True)[0]
        s_idx.append(int(s_logits.argmax()))
        t = int(model.scale_idx[g])
        s_margin = min(s_margin, float((s_logits[t] - torch.cat([s_logits[:t], s_logits[t + 1:]]).max()) / s_logits.abs().max().clamp_min(1e-30)))
    if margins is not None:
        margins["value"] = min_margin
        margins["scale"] = s_margin
    return out.view(model.R, d), s_idx, bad


@torch.no_grad()
def sample_text(model, prompt="\n", n=400, temp=0.8, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.tensor([model.encode_text(prompt)], device=model.T.device)
    for _ in range(n):
        logits = model(ids[:, -model.cfg.block_size:], head=slice(0, model.C), last=True)[0] / temp
        nxt = torch.multinomial(F.softmax(logits, -1).cpu(), 1, generator=g).to(ids.device)
        ids = torch.cat([ids, nxt[None]], dim=1)
    return model.decode_text(ids[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--data", default="data.noindex/shakespeare.txt")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval_n", type=int, default=32, help="validation batches for the LM loss")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    model, ck = load(args.ckpt, args.device)
    _, val_text, _ = load_text(args.data)
    val_ids = model.encode_corpus(val_text)
    seq = ck["args"]["seq"]
    results = {"params": model.n_params(), "rows": model.R, "d": model.cfg.d, "n_layer": model.cfg.n_layer,
               "scales": [float(v) for v in model.scale_values()], "generations": []}
    parent = model
    print(f"parent: {model.R} rows x {model.cfg.d} = {model.n_params():,} params, {model.cfg.n_layer} layers, "
          f"scales {[round(float(v), 4) for v in model.scale_values()]}; lm_val {eval_lm(model, val_ids, seq, 32, args.eval_n):.4f}", flush=True)
    print("--- parent sample ---\n" + sample_text(model) + "\n---------------------", flush=True)
    for g in range(1, args.generations + 1):
        t0 = time.time()
        margins = {}
        printed, s_idx, bad = self_print(parent, margins=margins)
        true = parent.codes().cpu()
        n_exact = int((printed == true).sum())
        exact = n_exact / printed.numel()
        rows_exact = float((printed == true).all(1).float().mean())
        child = TableGPT(parent.cfg)
        child.load_codes(printed, s_idx)
        child.frozen_scale = True  # a rebuilt model never re-estimates its scale
        child.to(args.device)
        child.eval()
        same = bool(torch.equal(printed, true)) and s_idx == parent.scale_idx.tolist()
        child_loss = eval_lm(child, val_ids, seq, 32, args.eval_n)
        rec = {"generation": g, "exact_weights": exact, "n_exact": n_exact, "n_weights": printed.numel(),
               "rows_exact": rows_exact, "scale_ok": s_idx == parent.scale_idx.tolist(),
               "bit_identical": same, "child_lm_val": child_loss, "bad_tokens": bad,
               "min_rel_margin_value": margins["value"], "rel_margin_scale": margins["scale"], "seconds": time.time() - t0}
        results["generations"].append(rec)
        print(f"gen {g}: exact {n_exact:,}/{printed.numel():,} rows_exact {rows_exact:.4f} scale_ok {rec['scale_ok']} "
              f"bit-identical={same} child lm_val {child_loss:.4f} bad_tokens {bad} "
              f"min rel margin {margins['value']:.3f} (scale {margins['scale']:.3f}) ({time.time() - t0:.0f}s)", flush=True)
        parent = child
    json.dump(results, open(os.path.join(os.path.dirname(args.ckpt), "verify.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
