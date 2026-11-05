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


def text_batch(ids, seq, batch, gen, device=None):
    ix = torch.randint(0, len(ids) - seq - 1, (batch,), generator=gen)
    x = torch.stack([ids[i:i + seq] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + seq] for i in ix])
    return x.to(device), y.to(device)


def quine_batch(model: TableGPT, rows: torch.Tensor, cols: torch.Tensor):
    """<Q> <row r> <col c>  ->  value token at the last position."""
    dev = model.T.device
    rows, cols = rows.to(dev), cols.to(dev)
    q = model.codes()[rows, cols]
    x = torch.stack([torch.full_like(rows, model.Q), rows, model.COORD0 + cols], dim=1)
    y = (q + model.VAL0 + model.cfg.k)[:, None]
    return x, y


def scale_batch(model: TableGPT, g: int = 0):
    """<Q> <SCALE_g> -> the scale token of group g; the loss is over that group's scale tokens only
    (what the printer decodes among)."""
    x = torch.tensor([model.scale_prompt(g)], device=model.T.device)
    y = torch.tensor([int(model.scale_idx[g])], device=model.T.device)
    return x, y


def scale_loss(model: TableGPT):
    tot = 0.0
    for g in range(model.cfg.n_scale):
        x, y = scale_batch(model, g)
        tot = tot + F.cross_entropy(model(x, head=model.scale_tokens(g), last=True), y)
    return tot


def scale_ok(model: TableGPT) -> bool:
    for g in range(model.cfg.n_scale):
        x, y = scale_batch(model, g)
        if int(model(x, head=model.scale_tokens(g), last=True).argmax(-1)) != int(y):
            return False
    return True


@torch.no_grad()
def eval_lm(model, ids, seq, batch, n=8):
    was = model.training
    model.eval()
    g = torch.Generator().manual_seed(1)
    tot = 0.0
    for _ in range(n):
        x, y = text_batch(ids, seq, batch, g, model.T.device)
        tot += F.cross_entropy(model(x, head=slice(0, model.C)).reshape(-1, model.C), y.reshape(-1)).item()
    model.train(was)
    return tot / n


@torch.no_grad()
def eval_quine(model, n=4096, gen=None):
    """Accuracy of the value answer (argmax restricted to value tokens) on random (row, col) pairs,
    plus the fraction of whole rows that are exactly right on a sample of 32 rows, and whether the
    scale token is right (argmax restricted to scale tokens)."""
    model.eval()
    d = model.cfg.d
    rows = torch.randint(0, model.R, (n,), generator=gen)
    cols = torch.randint(0, d, (n,), generator=gen)
    x, y = quine_batch(model, rows, cols)
    v0, v1 = model.VAL0, model.VAL0 + 2 * model.cfg.k + 1
    pred = model(x, head=slice(v0, v1), last=True).argmax(-1) + v0
    out = {"val_acc": float((pred == y[:, 0]).float().mean())}
    rows = torch.randint(0, model.R, (32,), generator=gen).repeat_interleave(d)
    cols = torch.arange(d).repeat(32)
    x, y = quine_batch(model, rows, cols)
    pred = model(x, head=slice(v0, v1), last=True).argmax(-1) + v0
    out["row_exact"] = float((pred == y[:, 0]).view(32, -1).all(1).float().mean())
    out["scale_ok"] = scale_ok(model)
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
    p.add_argument("--eval_n", type=int, default=8, help="validation batches per eval")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", default=None)
    p.add_argument("--init_circuit", action="store_true", help="start from the hand-installed copy circuit (frozen scale)")
    p.add_argument("--free_density", type=float, default=0.5, help="non-zero fraction at init in the free rows of the circuit model")
    p.add_argument("--protect", type=float, default=1.0, help="multiply updates to the circuit rows by this factor (<1 protects the hand-installed circuit)")
    p.add_argument("--binary", action="store_true", help="binary codes (+1/-1), no zeros")
    p.add_argument("--strict_causal", action="store_true", help="no self-attention (needed by the constructed circuit)")
    p.add_argument("--scale", type=float, default=0.0, help="pin and freeze the global scale at this value (0 = BitNet rule; with --init_circuit 0 = 1.0)")
    p.add_argument("--n_scale", type=int, default=1, help="2 = separate scales for the circuit rows and the free rows")
    p.add_argument("--free_scale", type=float, default=0.0, help="with --n_scale 2: pin the free rows' scale (0 = BitNet rule on the free rows)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)

    train_text, val_text, chars = load_text(args.data)
    cfg = TConfig(chars=chars, d=args.d, n_layer=args.n_layer, n_head=args.n_head, block_size=args.block, binary=args.binary,
                  strict_causal=args.strict_causal or args.init_circuit, n_scale=args.n_scale)
    model = TableGPT(cfg)
    if args.init_circuit:
        from .circuit import install_copy_circuit
        install_copy_circuit(model, scale=args.scale or 1.0, free_density=args.free_density,
                             free_scale=(args.free_scale or 0.15) if args.n_scale == 2 else None)
        if args.n_scale == 2 and args.free_scale == 0:
            model.frozen[1] = False  # the free rows' scale follows the BitNet rule
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        model.frozen_scale = bool(ck.get("frozen_scale", False))
    if args.scale > 0 and not args.init_circuit:
        # pin the global scale; rescale the shadow table so every code is preserved
        with torch.no_grad():
            old = float(model.scale_value()) if args.resume else 0.02
            model.scale_idx.fill_(int(torch.argmin((model.grid - args.scale).abs())))
            model.T.mul_(float(model.scale_value()) / old)
        model.frozen_scale = True
        print(f"scale pinned at {float(model.scale_value()):.3f} (was {old:.3f})", flush=True)
    if args.resume:
        model.frozen = list(ck.get("frozen", [model.frozen_scale] * model.cfg.n_scale))
    model.to(args.device)
    train_ids = torch.tensor(model.encode_text(train_text))
    val_ids = torch.tensor(model.encode_text(val_text))
    print(f"rows/tokens={model.R} params={model.n_params():,} device={args.device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))

    protected = None
    if args.protect < 1.0:
        from .circuit import circuit_mask
        protected = circuit_mask(model).to(args.device)
        print(f"protected entries: {int(protected.sum()):,} of {model.n_params():,}", flush=True)
    v0, v1 = model.VAL0, model.VAL0 + 2 * model.cfg.k + 1
    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        lam = args.lam if step >= args.lam_start else 0.0
        x, y = text_batch(train_ids, args.seq, args.batch, gen, args.device)
        logits = model(x, head=slice(0, model.C))  # text positions only ever predict characters
        lm_loss = F.cross_entropy(logits.reshape(-1, model.C), y.reshape(-1))
        rows = torch.randint(0, model.R, (args.qbatch,), generator=gen)
        cols = torch.randint(0, model.cfg.d, (args.qbatch,), generator=gen)
        qx, qy = quine_batch(model, rows, cols)
        # the answer is one of the value tokens, so the loss is over those tokens only (as the printer decodes)
        q_loss = F.cross_entropy(model(qx, head=slice(v0, v1), last=True), qy[:, 0] - v0)
        loss = lm_loss + lam * q_loss
        if step % 10 == 0:
            loss = loss + lam * scale_loss(model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if protected is not None:
            before = model.T.detach().clone()
        opt.step()
        if protected is not None:
            with torch.no_grad():
                model.T[protected] = before[protected] + args.protect * (model.T[protected] - before[protected])
        model.clamp_()
        if step % 50 == 0:
            print(f"step {step} lm {lm_loss.item():.3f} quine {q_loss.item():.3f} lam {lam:.1f} lr {lr_at(step):.1e} {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            codes = model.codes()
            rec = {"step": step, "lm_train": lm_loss.item(), "lm_val": eval_lm(model, val_ids, args.seq, args.batch, args.eval_n),
                   **eval_quine(model, gen=gen), "frac_zero": float((codes == 0).float().mean()),
                   "scale_idx": model.scale_idx.tolist(), "scales": [round(float(v), 4) for v in model.scale_values()],
                   "t": time.time() - t0}
            log.append(rec)
            print("EVAL " + json.dumps(rec), flush=True)
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"model": state, "cfg": cfg.__dict__, "args": vars(args), "frozen_scale": model.frozen_scale, "frozen": model.frozen},
                       os.path.join(args.out, "ckpt.pt"))
            json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
