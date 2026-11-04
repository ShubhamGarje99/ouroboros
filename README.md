# Ouroboros: a language model that prints its own weights

**Status (2026-11-05): done for v1. A 152,000-weight character model trained on Shakespeare (val loss 2.89 nats/char) prints all of its own weights exactly; the printout, loaded into an empty model, is bit-identical, and so is its own printout. Every number in this README is printed by a script in the repo.**

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

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu numpy matplotlib
curl -sL -o data.noindex/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
.venv/bin/python -m ouroboros.train_tm --steps 8000 --lam 1.0 --out runs.noindex/tm1
.venv/bin/python -m ouroboros.verify_tm runs.noindex/tm1/ckpt.pt --generations 3
```

Everything runs on a laptop CPU in minutes (115k parameters, ~11 steps/s on an M3).

## Results

All numbers below are produced by the scripts in this repo on a laptop CPU
(M3, 8 GB), with checkpoints under `runs.noindex/`. The model is 2,375 rows × 64
= 152,000 ternary weights, 2 layers, 4 heads, context 128.

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

Figures: `figures/training.png` (self-print accuracy, LM loss and zero fraction;
the tm18 curve shows the language model learning while self-print stays at 100 %)
and `figures/generations.png`.

## One line for a resume

Built Ouroboros, the first transformer that prints its own weights exactly: a
152k-parameter character LM whose vocabulary is its own weight rows; proved
memorisation cannot work, hand-built and verified a bit-exact self-copy circuit,
and trained the LM on top of it without losing exactness (CPU only).

## Related work

- Chang & Lipson, *Neural Network Quine*, 2018 (arXiv:1803.05859): MLP that regresses its own weights; ~90% MNIST as an auxiliary task.
- Schmidgall, *Self-Replicating Neural Programs*, 2021 (arXiv:2109.12786): replication of a training program in a toy language.
- BitNet b1.58 (Ma et al., 2024): ternary-weight language models, the reason k = 1 is a sane choice.
- Allen-Zhu & Li, *Physics of Language Models 3.3* (2024): ~2 bits of knowledge per parameter, the capacity bound that makes self-printing non-trivial.
