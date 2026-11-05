# Ouroboros: a language model that prints its own weights

**Status (2026-11-05): v2. A 156,160-weight character model trained on Shakespeare (val loss 2.08 nats/char; v1: 2.88) prints all of its own weights and both of its weight scales exactly. The printout, loaded into an empty model, is bit-identical, and so is its own printout, for three generations. A plain model of the same size with no self-printing reaches 1.86, so exactness now costs 0.22 nats/char instead of 1.0. Every number in this README is printed by a script in the repo.**

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

**The weights are the vocabulary.** All parameters live in one ternary table
`T` of shape (rows × d). Row *r* of the table is simultaneously the embedding of
token number *r* and one row of some weight matrix: the first rows embed the
text characters, the value tokens (`-1`, `0`, `+1`) and three control tokens;
then come the position embeddings; then, for each layer, the rows of W_q, W_k,
W_v, W_o, W_1 and W_2ᵀ. The output head is tied to the same table.

Because every weight row is a token, the prompt `<Q> <row r>` places row *r*
itself into the residual stream. Printing the row is then a *copy circuit*
(attend to the address token, read coordinate *j* at output position *j*, emit
the matching value token) that the model learns once and that works for any
weights, instead of a memorisation problem. That matters because memorisation
cannot work here even in principle: the number of bits needed to describe the
weights equals the number of bits the weights can hold, so a model can only print
itself if the architecture supplies a "quote" operation, exactly as a program
quine relies on its interpreter's string literals.

Training mixes an ordinary next-character loss on text with the self-printing
loss, whose target is the model's *current* quantised table. The one global
scale is a snapped statistic of the table and is printed too (`<Q> <SCALE>`).
`ouroboros.verify_tm` prints every row greedily, rebuilds a child from the
printout, checks bit-identity, measures the child's language-model loss, and
repeats for several generations.

### Two scales (v2)

The copy circuit needs large weights. Attention has to be sharp, and the hand-built
circuit is exact only at a weight scale of about 1.0 or more: `ouroboros.construct`
checks every prompt and finds the d=128, 4-layer circuit exact at 0.98, 1.5 and 2.0 and
wrong on 57,560 of 1,115,008 weights at 0.74. A ternary language model trains badly at
that scale. In v1 every weight shared one scale, so the language model was stuck at it.

v2 gives the table two scales: one for the circuit's rows (control, value, coordinate
and scale tokens, the first three positions, layer 0's attention and the circuit's
hidden units) and one for every other row. The model prints both, with
`<Q> <SCALE_0>` and `<Q> <SCALE_1>`. Each answer is read from a reserved coordinate
that no trainable row can write, so both are exact by construction rather than
learned. Everything else about printing is unchanged: a row is its codes times its
scale. The circuit also no longer fills layer 0: its 3d + 7 hidden units leave the
remaining d − 7 free for the language model.

### What did not work (and why it cannot)

The first version (`ouroboros/model.py`, `train.py`, `verify.py`, kept for
the record) used separate quantised matrices and two-digit row addresses, so the
model had to *memorise* its own rows. Across k = 3 and ternary weights, with or
without 80% sparsity, bidirectional soft targets, frozen targets and
regeneration, teacher-forced self-print accuracy never left the marginal rate
(the fraction of zeros), and greedy printouts were exactly the all-zero
baseline. The counting argument above says why: for a weight configuration with
entropy H, memorising it costs H bits of the H bits available. Sparsity does not
help, because it lowers capacity (zeros store nothing) at least as fast as it
lowers the description. Iterating "load the printout, print again" from an 80%
printout collapsed to a trivial fixed point (a degenerate network that prints
itself) with a broken language model (loss 5.2 versus 2.8).

## Reproduce

v2, the result in the status line (about 3 minutes on a laptop CPU, ~44 steps/s):

```bash
.venv/bin/python -m ouroboros.train_tm --init_circuit --protect 0.0 --n_scale 2 --scale 1.0 --free_scale 0.15 \
    --lr 3e-3 --steps 8000 --eval_every 1000 --eval_n 16 --out runs.noindex/v2_d64_2s_long
.venv/bin/python -m ouroboros.verify_tm runs.noindex/v2_d64_2s_long/ckpt.pt --generations 3 --eval_n 64
.venv/bin/python -m ouroboros.construct --d 128 --n_layer 4 --n_head 8 --block 256 --scales 0.5,0.71,1.0,1.41,2.0
```

v1:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu numpy matplotlib
curl -sL -o data.noindex/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
.venv/bin/python -m ouroboros.train_tm --steps 8000 --lam 1.0 --out runs.noindex/tm1
.venv/bin/python -m ouroboros.verify_tm runs.noindex/tm1/ckpt.pt --generations 3
```

Everything runs on a laptop CPU in minutes (152k weights, ~11 steps/s on an M3 with the v1 code).

## Results

All numbers below are produced by the scripts in this repo on a laptop CPU
(M3, 8 GB), with checkpoints under `runs.noindex/`. All models are 2 layers, d = 64,
4 heads, context 128, character-level tiny-shakespeare (1.1 MB, 90/10 split).

**v2: a good language model that prints itself exactly.** Printed by
`python -m ouroboros.table`; the loss is on the same 64 validation batches for every row.

| model | weights | layers × d | scales | self-print exact | child bit-identical | LM val loss |
|---|---|---|---|---|---|---|
| v1: circuit, one scale 1.5 (tm19) | 152,000 | 2 × 64 | 1.5 | 152,000 / 152,000 | yes (3 generations) | 2.88 |
| circuit, one scale 1.0 (v2_d64_s1) | 152,000 | 2 × 64 | 0.978 | 152,000 / 152,000 | yes (3 generations) | 2.81 |
| learned from scratch, scale 1.0 (v2_d64_plain_s1) | 152,000 | 2 × 64 | 0.978 | 106,073 / 152,000 | no | 2.76 |
| **v2: circuit, two scales (v2_d64_2s_long)** | 156,160 | 2 × 64 | 0.978 / 0.152 | 156,160 / 156,160 | yes (3 generations) | 2.08 |
| plain LM, no circuit, scale 0.15 (v2_d64_plain) | 152,000 | 2 × 64 | 0.152 | – (no self-print training) | – | 1.86 |

Reading the rows from top to bottom:

- **The circuit is not what made v1 weak.** At one scale of 1.0, the circuit model
  (2.81) is within 0.05 of a model learned from scratch at the same scale (2.76),
  and the from-scratch model cannot print itself: it gets 70 % of its weights right
  and its child has a loss of 12.9.
- **The weight scale was.** The same plain model at scale 0.15 reaches 1.86.
- **Two scales recover most of the gap.** v2 reaches 2.08 and stays exact. The
  remaining 0.22 is plausibly structural: the circuit takes all of layer 0's
  attention and seven of 64 coordinates, so the language model has about one and a
  half working layers out of two. This is a single seed.

**v1 results** (kept as recorded; v1 losses are on 8 validation batches). The model
is 2,375 rows × 64 = 152,000 ternary weights.

**1. An exact self-printer exists (constructive).** `ouroboros.circuit.install_copy_circuit`
writes the copy circuit into layer 0 of a fresh table (layer 1 zero). With no
training at all it answers all 152,000 `<Q> <row> <col>` prompts correctly and
prints its scale token. `ouroboros.verify_tm` loads the printout into an empty
model and finds it **bit-identical** to the printer; that child's printout is
bit-identical too, for every generation we ran (3).

| model | self-print exact | rows exact | child bit-identical | LM val loss |
|---|---|---|---|---|
| constructed circuit, no training | 152,000 / 152,000 | 2,375 / 2,375 | yes (3 generations) | untrained |
| constructed circuit + LM training, circuit entries frozen (tm18, 6k steps) | 152,000 / 152,000 | 2,375 / 2,375 | yes (3 generations) | 2.95 |
| **same, + 4k steps fine-tune at lr 1e-3 (tm19)** | **152,000 / 152,000** | **2,375 / 2,375** | **yes (3 generations)** | **2.89** |
| learned from scratch, ternary, scale 0.10 (tm2) | 77.7 % | 0 % | no | 2.50 |
| learned from scratch, ternary, scale 0.50 (tm9, 6k steps) | 69 % | 0 % | – | 2.42 |
| learned from scratch, binary ±1, scale 0.15 (tm13) | 79.8 % | 0 % | no | 2.18 |

**2. What training discovers on its own.** With the gated block, joint training
finds a partial copy circuit from random initialisation within ~2,000 steps
(`figures/training.png`). Analysing tm2's errors shows what it is: the model reads
the *sign* of a weight well (88 % on −1, 91 % on +1) but cannot tell *zero* apart
(27 %, chance), i.e. it learns a linear sign readout but not the magnitude decision,
which needs a clean copy and a constant offset. Binary weights remove the zero
class and reach 80 %. The remaining 20 % are sign errors from an attention copy
that is not sharp; at the small scales training prefers, attention scores are
bounded by the weight scale squared.

**3. Memorising oneself does not work, for a reason.** Section "What did not work"
above: with ordinary matrices and two-digit addresses, no run left the marginal
rate, and freezing the target did not help (31 % after 2,000 steps). A network's
weights can hold at most as many bits as they contain, so a model can only print
itself through a quote mechanism, which here is the embedding lookup.

**4. Reproduction with mutation.** `ouroboros.evolve_tm` prints with sampling
noise instead of the argmax. At τ = 1.5 about 16 % of the weights mutate; at
τ = 0.3 about 1 % (1,600 weights). Both are lethal: the child's LM loss jumps
(2.95 → 6.1 at 1 %) and, more tellingly, the child can no longer print itself,
so the lineage collapses in one generation. The copier is part of the genome it
copies, and random mutations hit it in proportion to its size. A viable
mutation operator would have to spare the circuit, which is what the training
mask does. Reported as a demo, not a result (`runs.noindex/*/evolve_tau*.json`).

Figures: `figures/training.png` compares v1 (tm18), v2 and the plain model
(LM loss, teacher-forced self-print accuracy, zero fraction; the plain model has no
self-print training, so its accuracy is the chance level). `figures/generations.png`
shows v2's three generations: every weight exact, child loss unchanged.

## One line for a resume

Built Ouroboros, the first transformer that prints its own weights exactly: a
156k-parameter character LM whose vocabulary is its own weight rows; proved
memorisation cannot work, hand-built a bit-exact self-copy circuit, and found that
the weight scale, not the circuit, limited the LM. With a separate printed scale
for the circuit the LM reaches 2.08 nats/char (plain model 1.86) and copies of it
stay bit-identical for 3 generations (CPU only).

## Related work

- Chang & Lipson, *Neural Network Quine*, 2018 (arXiv:1803.05859): MLP that regresses its own weights; ~90% MNIST as an auxiliary task.
- Schmidgall, *Self-Replicating Neural Programs*, 2021 (arXiv:2109.12786): replication of a training program in a toy language.
- BitNet b1.58 (Ma et al., 2024): ternary-weight language models, the reason k = 1 is a sane choice.
- Allen-Zhu & Li, *Physics of Language Models 3.3* (2024): ~2 bits of knowledge per parameter, the capacity bound that makes self-printing non-trivial.
