"""Build the hand-installed circuit for a given architecture, check it at several weight scales
(every one of the R*d prompts, plus the scale prompt), report the smallest logit margin, and save
the checkpoint for one scale so `train_tm --resume` / `verify_tm` can use it."""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .circuit import install_copy_circuit
from .tm import n_circuit_units
from .data import load_text
from .tm import TableGPT, TConfig
from .verify_tm import self_print


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.noindex/shakespeare.txt")
    p.add_argument("--out", default="runs.noindex/constructed")
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--scales", default="1.5")
    p.add_argument("--save_scale", type=float, default=None, help="save the checkpoint built at this scale (default: last)")
    p.add_argument("--free_density", type=float, default=0.5)
    p.add_argument("--n_scale", type=int, default=1)
    p.add_argument("--free_scale", type=float, default=0.15)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    _, _, chars = load_text(args.data)
    cfg = TConfig(chars=chars, d=args.d, n_layer=args.n_layer, n_head=args.n_head, block_size=args.block, strict_causal=True, n_scale=args.n_scale)
    os.makedirs(args.out, exist_ok=True)
    report = {"d": args.d, "n_layer": args.n_layer, "n_head": args.n_head, "block": args.block, "scales": []}
    for s in [float(v) for v in args.scales.split(",")]:
        model = TableGPT(cfg)
        install_copy_circuit(model, scale=s, free_density=args.free_density, free_scale=args.free_scale if args.n_scale == 2 else None)
        model.to(args.device).eval()
        t0 = time.time()
        margins = {}
        printed, s_idx, bad = self_print(model, margins=margins)
        true = model.codes().cpu()
        n_exact = int((printed == true).sum())
        rec = {"scales": [float(v) for v in model.scale_values()], "n_exact": n_exact, "n_weights": printed.numel(),
               "rows_exact": float((printed == true).all(1).float().mean()), "scale_ok": s_idx == model.scale_idx.tolist(),
               "min_rel_margin_value": margins["value"], "rel_margin_scale": margins["scale"], "bad_tokens": bad,
               "seconds": time.time() - t0}
        report["scales"].append(rec)
        print(f"scales {[round(v, 4) for v in rec['scales']]}: exact {n_exact:,}/{printed.numel():,} rows_exact {rec['rows_exact']:.4f} "
              f"scale_ok {rec['scale_ok']} min rel margin {margins['value']:.3f} (scale {margins['scale']:.3f}) "
              f"bad_tokens {bad} ({rec['seconds']:.0f}s)", flush=True)
        if args.save_scale is None or abs(args.save_scale - s) < 1e-6:
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"model": state, "cfg": cfg.__dict__, "frozen_scale": True,
                        "args": {"seq": args.block, "d": args.d, "n_layer": args.n_layer, "n_head": args.n_head,
                                 "block": args.block, "init_circuit": True, "scale": s, "n_scale": args.n_scale,
                                 "free_scale": args.free_scale}, "frozen": model.frozen},
                       os.path.join(args.out, "ckpt.pt"))
    report["rows"], report["params"], report["circuit_units"] = model.R, model.n_params(), n_circuit_units(args.d)
    report["free_units_layer0"] = 4 * args.d - n_circuit_units(args.d)
    json.dump(report, open(os.path.join(args.out, "construct.json"), "w"), indent=1)
    print(f"rows {model.R} params {model.n_params():,} circuit units {n_circuit_units(args.d)} of {4 * args.d}")


if __name__ == "__main__":
    main()
