from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from .data import QuineBatches, TextBatches, load_text
from .model import GPT, Config
from .vocab import Vocab


def soft_quine_loss(logits, y, mask, soft, vocab):
    """CE against the soft target on value positions, plus hard CE on the <END> position."""
    logp = torch.log_softmax(logits, dim=-1)  # full vocabulary, so mass on non-value tokens is penalised
    cols = soft.shape[1]
    lp_vals = logp[:, Vocab.PREFIX_LEN - 1: Vocab.PREFIX_LEN - 1 + cols, vocab.VAL0: vocab.VAL0 + 2 * vocab.k + 1]
    ce_vals = -(soft * lp_vals).sum(-1).mean()
    ce_end = -logp[:, -1, vocab.END].mean()
    return ce_vals + ce_end / cols


@torch.no_grad()
def regenerate(model, qb: QuineBatches, conf: float, batch_rows: int = 128) -> dict:
    """Overwrite weights with the model's own confident predictions about them.

    For every row of every matrix, run the self-print prompt teacher-forced,
    take the restricted argmax over value tokens and its probability, and
    where the model is at least `conf` sure of a value that differs from the
    stored code, set the shadow weight to that value.  Returns counts.
    """
    model.eval()
    v0, v1 = qb.vocab.VAL0, qb.vocab.VAL0 + 2 * qb.vocab.k + 1
    changed = total = 0
    for name, mat in qb.mats:
        s = mat.scale_value()
        for r0 in range(0, mat.rows, batch_rows):
            rows = torch.arange(r0, min(mat.rows, r0 + batch_rows))
            x, y, mask = qb.rows_of(name, mat, rows)
            logits = model(x)[:, Vocab.PREFIX_LEN - 1: Vocab.PREFIX_LEN - 1 + mat.cols, v0:v1]
            p = torch.softmax(logits, -1)
            pconf, pred = p.max(-1)
            pred = pred - qb.vocab.k  # back to integer codes
            cur = mat.q()[rows]
            flip = (pconf >= conf) & (pred != cur)
            if flip.any():
                w = mat.weight[rows]
                w[flip] = pred[flip].to(torch.float32) * s
                mat.weight[rows] = w
                changed += int(flip.sum())
            total += cur.numel()
    model.train()
    return {"regen_changed": changed, "regen_frac": changed / total}


def masked_ce(logits, y, mask):
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    loss = loss.view_as(y)[mask]
    return loss.mean()


def build(args):
    train_text, val_text, chars = load_text(args.data)
    # the vocab depends on the matrix registry, which depends only on n_layer / tie_head
    probe = GPT(Config(vocab_size=1, d=8, n_layer=args.n_layer, n_head=1, k=args.k, tie_head=not args.untied))
    mat_names = [n for n, _ in probe.matrices()]
    vocab = Vocab(chars, args.k, mat_names)
    cfg = Config(vocab_size=vocab.size, block_size=args.block, d=args.d, n_layer=args.n_layer,
                 n_head=args.n_head, k=args.k, tie_head=not args.untied, sparsity=args.sparsity)
    model = GPT(cfg)
    return model, vocab, cfg, train_text, val_text


@torch.no_grad()
def eval_lm(model, batches, n=10):
    model.eval()
    losses = []
    for _ in range(n):
        x, y = batches()
        losses.append(F.cross_entropy(model(x).reshape(-1, model.cfg.vocab_size), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def eval_quine_tf(model, qb: QuineBatches, n_rows=256, gen=None):
    """Teacher-forced accuracy on random rows: per-value accuracy and exact-row fraction."""
    model.eval()
    tok_correct = tok_total = rows_exact = rows_total = 0
    for name, mat in qb.mats:
        take = max(1, round(n_rows * mat.rows / sum(m.rows for _, m in qb.mats)))
        rows = torch.randint(0, mat.rows, (min(take, mat.rows),), generator=gen)
        x, y, mask = qb.rows_of(name, mat, rows)
        logits = model(x)
        pred = logits.argmax(-1)
        # at value positions the printer only ever chooses among value tokens; measure the same way
        v0, v1 = qb.vocab.VAL0, qb.vocab.VAL0 + 2 * qb.vocab.k + 1
        pred_val = logits[:, :, v0:v1].argmax(-1) + v0
        pred = torch.where((y >= v0) & (y < v1), pred_val, pred)
        ok = (pred == y) | ~mask
        tok_correct += int(((pred == y) & mask).sum())
        tok_total += int(mask.sum())
        rows_exact += int(ok.all(dim=1).sum())
        rows_total += len(rows)
    x, y, mask = qb.scale_batch()
    pred = model(x).argmax(-1)
    scale_ok = float(((pred == y) | ~mask).all(dim=1).float().mean())
    model.train()
    return {"val_acc": tok_correct / tok_total, "row_exact": rows_exact / rows_total, "scale_exact": scale_ok}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.noindex/shakespeare.txt")
    p.add_argument("--out", default="runs.noindex/run")
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--block", type=int, default=288)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--untied", action="store_true")
    p.add_argument("--sparsity", type=float, default=0.0, help="target fraction of zero codes")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--qbatch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--lam", type=float, default=1.0, help="weight of the self-printing loss")
    p.add_argument("--frozen_target", action="store_true", help="diagnostic: memorise a snapshot of the codes taken at lam_start")
    p.add_argument("--regen_every", type=int, default=0, help="every N steps overwrite weights with confident self-predictions")
    p.add_argument("--regen_conf", type=float, default=0.9)
    p.add_argument("--regen_start", type=int, default=0)
    p.add_argument("--bidir", type=float, default=0.0, help="soft-target width tau (0 = hard targets)")
    p.add_argument("--lam_warmup", type=int, default=0, help="steps over which lam ramps 0->lam")
    p.add_argument("--lam_start", type=int, default=0, help="step at which the quine loss switches on")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)

    model, vocab, cfg, train_text, val_text = build(args)
    if args.resume:
        model.load_state_dict(torch.load(args.resume)["model"])
    print(f"vocab={vocab.size} params={model.n_params():,} matrices={len(model.matrices())} "
          f"rows={sum(m.rows for _, m in model.matrices())}", flush=True)

    tb = TextBatches(train_text, vocab, args.seq, args.batch, gen)
    vb = TextBatches(val_text, vocab, args.seq, args.batch, torch.Generator().manual_seed(1))
    qb = QuineBatches(model, vocab, args.qbatch, gen)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))

    log = []
    t0 = time.time()
    prev_codes = {n: q.clone() for n, (q, _) in model.codes().items()}
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        if args.frozen_target and step == args.lam_start and qb.frozen is None:
            qb.frozen = {n: q.clone() for n, (q, _) in model.codes().items()}
            print("froze quine targets", flush=True)
        if step < args.lam_start:
            lam = 0.0
        elif args.lam_warmup:
            lam = args.lam * min(1.0, (step - args.lam_start) / args.lam_warmup)
        else:
            lam = args.lam

        x, y = tb()
        lm_loss = F.cross_entropy(model(x).reshape(-1, cfg.vocab_size), y.reshape(-1))
        if args.bidir > 0:
            qx, qy, qm, qsoft = qb(args.bidir)
            q_loss = soft_quine_loss(model(qx), qy, qm, qsoft, vocab)
        else:
            qx, qy, qm = qb()
            q_loss = masked_ce(model(qx), qy, qm)
        loss = lm_loss + lam * q_loss
        if step % 10 == 0:
            sx, sy, sm = qb.scale_batch()
            loss = loss + lam * masked_ce(model(sx), sy, sm)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.clamp_()
        if args.regen_every and step >= args.regen_start and step % args.regen_every == 0:
            rg = regenerate(model, qb, args.regen_conf)
            print(f"regen step {step}: changed {rg['regen_changed']} ({100 * rg['regen_frac']:.2f}%)", flush=True)

        if step % 50 == 0:
            print(f"step {step} lm {lm_loss.item():.3f} quine {q_loss.item():.3f} lam {lam:.2f} "
                  f"lr {lr_at(step):.2e} {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            cur = model.codes()
            churn = sum(int((cur[n][0] != prev_codes[n]).sum()) for n in prev_codes) / model.n_params()
            prev_codes = {n: q.clone() for n, (q, _) in cur.items()}
            rec = {"step": step, "lm_train": lm_loss.item(), "lm_val": eval_lm(model, vb),
                   **eval_quine_tf(model, qb, gen=gen), **model.code_stats(), "churn": churn, "t": time.time() - t0}
            log.append(rec)
            print("EVAL " + json.dumps(rec), flush=True)
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "chars": vocab.chars,
                        "mat_names": vocab.mat_names, "args": vars(args)}, os.path.join(args.out, "ckpt.pt"))
            json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
