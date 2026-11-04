from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from .data import load_text
from .tm import TableGPT, TConfig


def text_batch(ids, seq, batch, gen):
    ix = torch.randint(0, len(ids) - seq - 1, (batch,), generator=gen)
    return torch.stack([ids[i:i + seq] for i in ix]), torch.stack([ids[i + 1:i + 1 + seq] for i in ix])


def quine_batch(model: TableGPT, rows: torch.Tensor, cols: torch.Tensor):
    """<Q> <row r> <col c>  ->  value token at the last position."""
    q = model.codes()[rows, cols]
    x = torch.stack([torch.full_like(rows, model.Q), rows, model.COORD0 + cols], dim=1)
    y = (q + model.VAL0 + model.cfg.k)[:, None]
    return x, y


def scale_batch(model: TableGPT):
    seq = torch.tensor([[model.Q, model.SCALE_ADDR, model.SCALE0 + int(model.scale_idx), model.END]])
    x, y = seq[:, :-1], seq[:, 1:]
    mask = torch.tensor([[False, True, True]])
    return x, y, mask


def masked_ce(logits, y, mask):
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    return loss.view_as(y)[mask].mean()


@torch.no_grad()
def eval_lm(model, ids, seq, batch, n=8):
    model.eval()
    g = torch.Generator().manual_seed(1)
    tot = 0.0
    for _ in range(n):
        x, y = text_batch(ids, seq, batch, g)
        tot += F.cross_entropy(model(x)[..., : model.C].reshape(-1, model.C), y.reshape(-1)).item()
    model.train()
    return tot / n


@torch.no_grad()
def eval_quine(model, n=4096, gen=None):
    """Accuracy of the value answer (argmax restricted to value tokens) on random (row, col) pairs,
    plus the fraction of whole rows that are exactly right on a sample of 32 rows."""
    model.eval()
    rows = torch.randint(0, model.R, (n,), generator=gen)
    cols = torch.randint(0, model.cfg.d, (n,), generator=gen)
    x, y = quine_batch(model, rows, cols)
    v0, v1 = model.VAL0, model.VAL0 + 2 * model.cfg.k + 1
    pred = model(x)[:, -1, v0:v1].argmax(-1) + v0
    out = {"val_acc": float((pred == y[:, 0]).float().mean())}
    rows = torch.randint(0, model.R, (32,), generator=gen).repeat_interleave(model.cfg.d)
    cols = torch.arange(model.cfg.d).repeat(32)
    x, y = quine_batch(model, rows, cols)
    pred = model(x)[:, -1, v0:v1].argmax(-1) + v0
    out["row_exact"] = float((pred == y[:, 0]).view(32, -1).all(1).float().mean())
    x, y, mask = scale_batch(model)
    out["scale_ok"] = bool(((model(x).argmax(-1) == y) | ~mask).all())
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.noindex/shakespeare.txt")
    p.add_argument("--out", default="runs.noindex/tm")
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--qbatch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--lam_start", type=int, default=0)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--resume", default=None)
    p.add_argument("--init_circuit", action="store_true", help="start from the hand-installed copy circuit (frozen scale)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)

    train_text, val_text, chars = load_text(args.data)
    cfg = TConfig(chars=chars, d=args.d, n_layer=args.n_layer, n_head=args.n_head, block_size=args.block)
    model = TableGPT(cfg)
    if args.init_circuit:
        from .circuit import install_copy_circuit
        install_copy_circuit(model)
    if args.resume:
        model.load_state_dict(torch.load(args.resume)["model"])
        model.frozen_scale = bool(torch.load(args.resume).get("frozen_scale", False))
    train_ids = torch.tensor(model.encode_text(train_text))
    val_ids = torch.tensor(model.encode_text(val_text))
    print(f"rows/tokens={model.R} params={model.n_params():,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))

    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        lam = args.lam if step >= args.lam_start else 0.0
        x, y = text_batch(train_ids, args.seq, args.batch, gen)
        logits = model(x)[..., : model.C]  # text positions only ever predict characters
        lm_loss = F.cross_entropy(logits.reshape(-1, model.C), y.reshape(-1))
        rows = torch.randint(0, model.R, (args.qbatch,), generator=gen)
        cols = torch.randint(0, model.cfg.d, (args.qbatch,), generator=gen)
        qx, qy = quine_batch(model, rows, cols)
        q_loss = F.cross_entropy(model(qx)[:, -1], qy[:, 0])
        loss = lm_loss + lam * q_loss
        if step % 10 == 0:
            sx, sy, sm = scale_batch(model)
            loss = loss + lam * masked_ce(model(sx), sy, sm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.clamp_()
        if step % 50 == 0:
            print(f"step {step} lm {lm_loss.item():.3f} quine {q_loss.item():.3f} lam {lam:.1f} lr {lr_at(step):.1e} {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            codes = model.codes()
            rec = {"step": step, "lm_train": lm_loss.item(), "lm_val": eval_lm(model, val_ids, args.seq, args.batch),
                   **eval_quine(model, gen=gen), "frac_zero": float((codes == 0).float().mean()),
                   "scale_idx": int(model.scale_idx), "t": time.time() - t0}
            log.append(rec)
            print("EVAL " + json.dumps(rec), flush=True)
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "args": vars(args), "frozen_scale": model.frozen_scale},
                       os.path.join(args.out, "ckpt.pt"))
            json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
