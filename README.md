# Ouroboros: a language model that prints its own weights

**Status: work in progress (2026-11-05). Numbers below are filled in from `runs.noindex/*/verify.json` as runs finish.**

Ouroboros is a small character-level GPT with one unusual ability: ask it for
any row of any of its own weight matrices and it answers with the exact
integers stored there. Print every row, load the printout into an empty copy of
the architecture, and you get the same model back, bit for bit. The copy can
then print itself again.

As far as we can tell this is the first transformer, and the first language
model of any kind, that reproduces its own parameters. The only prior neural
network quine (Chang & Lipson, 2018) was a feed-forward regressor that
approximated its ~21k continuous weights and did not reproduce them exactly.

## How it works

1. **Every parameter is a small integer.** Each weight matrix is stored as
   integers in `[-k, k]` times one per-matrix scale that lives on a 64-level
   log grid. Training uses a straight-through estimator, so the model that is
   trained is the quantized model, and "the weights" are a finite string of
   symbols that can be spoken.
2. **Every row has an address.** The vocabulary contains, besides the text
   characters, one token per weight value, one token per scale level, two
   base-32 digits for row numbers, and one token per matrix. The prompt
   `<Q> <MAT_L1.w1> <3> <17>` means "row 3·32+17 of the first MLP matrix of
   layer 1", and the model answers with that row's integers followed by
   `<END>`.
3. **Joint training toward a fixed point.** Each step mixes an ordinary
   next-character loss on text with a *self-printing* loss whose target is the
   model's *current* quantized weights. The target moves as the weights move;
   training converges when the weights both model text and describe themselves.
4. **Verification.** `ouroboros.verify` prints every row greedily, rebuilds a
   child from the printout, checks bit-identity with the parent, measures the
   child's language-model loss, and repeats for several generations.
5. **Reproduction with mutation.** `ouroboros.evolve` prints stochastically
   (sampling value tokens at temperature τ) to produce mutated offspring,
   selects on validation loss, and iterates. No gradients anywhere.

## Why the fixed point is not free

An N-parameter network can memorise on the order of 2 bits per parameter of
arbitrary data, but describing itself takes N·log2(2k+1) bits. With k = 3 that
is 2.8 bits per weight, more than the capacity. The model can only print itself
if its weights are *structured* (sparse, repetitive, predictable from the
address and the preceding values), and the self-printing loss is exactly the
pressure that makes them so. The experiments below measure where that trade-off
lands for ternary (k = 1) and 7-level (k = 3) weights.

## Reproduce

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu numpy matplotlib
curl -sL -o data.noindex/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
.venv/bin/python -m ouroboros.train --k 1 --steps 10000 --lam_start 1500 --lam_warmup 1000 --out runs.noindex/k1
.venv/bin/python -m ouroboros.verify runs.noindex/k1/ckpt.pt --generations 3
.venv/bin/python -m ouroboros.evolve runs.noindex/k1/ckpt.pt --tau 1.0
```

Everything runs on a laptop CPU in minutes (125k parameters, ~25 steps/s on an M3).

## Results

_(pending)_

## Related work

- Chang & Lipson, *Neural Network Quine*, 2018 (arXiv:1803.05859): MLP that regresses its own weights; ~90% MNIST as an auxiliary task.
- Schmidgall, *Self-Replicating Neural Programs*, 2021 (arXiv:2109.12786): replication of a training program in a toy language.
- BitNet b1.58 (Ma et al., 2024): ternary-weight language models, the reason k = 1 is a sane choice.
- Allen-Zhu & Li, *Physics of Language Models 3.3* (2024): ~2 bits of knowledge per parameter, the capacity bound that makes self-printing non-trivial.
