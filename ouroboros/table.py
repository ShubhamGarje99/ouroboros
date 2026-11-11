"""Print the README results rows from checkpoints and their verify.json files.

For each run: parameter count, layers, scales, LM val loss on the same 64 validation batches,
and (if verify.json exists) the self-print result of every generation."""
from __future__ import annotations

import argparse
import json
import os

import torch

from .data import load_text
from .train_tm import eval_lm
from .verify_tm import load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="label=runs.noindex/<run>")
    p.add_argument("--data", default="data.noindex/shakespeare.txt")
    p.add_argument("--eval_n", type=int, default=64)
    p.add_argument("--threads", type=int, default=3)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    _, val_text, _ = load_text(a.data)
    print("| model | weights | layers × d | scales | self-print exact | child bit-identical | LM val loss |")
    print("|---|---|---|---|---|---|---|")
    for spec in a.runs:
        label, run = spec.split("=", 1)
        model, ck = load(os.path.join(run, "ckpt.pt"))
        loss = eval_lm(model, model.encode_corpus(val_text), ck["args"]["seq"], 32, a.eval_n)
        scales = " / ".join(f"{float(s):.3g}" for s in model.scale_values())
        vp = os.path.join(run, "verify.json")
        if os.path.exists(vp):
            gens = json.load(open(vp))["generations"]
            n = gens[0].get("n_exact", round(gens[0]["exact_weights"] * model.n_params()))
            exact = f"{n:,} / {model.n_params():,}"
            ident = f"yes ({len(gens)} generation{'s' if len(gens) > 1 else ''})" if all(g["bit_identical"] for g in gens) else "no"
        elif not ck["args"].get("init_circuit"):
            exact, ident = "– (no self-print training)" if ck["args"].get("lam", 1.0) == 0 else "not verified", "–"
        else:
            exact, ident = "not verified", "–"
        print(f"| {label} | {model.n_params():,} | {model.cfg.n_layer} × {model.cfg.d} | {scales} | {exact} | {ident} | {loss:.2f} |")


if __name__ == "__main__":
    main()
