"""Figures from training logs and verify/evolve reports."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training(log_paths: dict[str, str], out: str):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for label, path in log_paths.items():
        log = json.load(open(path))
        steps = [r["step"] for r in log]
        ax[0].plot(steps, [r["lm_val"] for r in log], label=label)
        ax[1].plot(steps, [100 * r["val_acc"] for r in log], label=label)
        ax[2].plot(steps, [100 * r["frac_zero"] for r in log], label=label)
    ax[0].set_title("language-model val loss (nats/char)")
    ax[1].set_title("self-print accuracy, teacher-forced (%)")
    ax[2].set_title("weights that are exactly zero (%)")
    for a in ax:
        a.set_xlabel("step"); a.grid(alpha=.3)
    ax[0].legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_generations(verify_path: str, out: str):
    rep = json.load(open(verify_path))
    gens = rep["generations"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    x = [g["generation"] for g in gens]
    ax[0].plot(x, [100 * g["exact_weights"] for g in gens], "o-")
    ax[0].set_ylabel("weights reproduced exactly (%)"); ax[0].set_xlabel("generation")
    ax[1].plot(x, [g["child_lm_val"] for g in gens], "o-")
    ax[1].set_ylabel("child LM val loss"); ax[1].set_xlabel("generation")
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs="*", default=[], help="label=path/to/log.json")
    p.add_argument("--verify", default=None)
    p.add_argument("--outdir", default="figures")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.logs:
        plot_training(dict(s.split("=", 1) for s in a.logs), os.path.join(a.outdir, "training.png"))
    if a.verify:
        plot_generations(a.verify, os.path.join(a.outdir, "generations.png"))
