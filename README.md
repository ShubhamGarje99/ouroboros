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

_(pending)_

## Related work

- Chang & Lipson, *Neural Network Quine*, 2018 (arXiv:1803.05859): MLP that regresses its own weights; ~90% MNIST as an auxiliary task.
- Schmidgall, *Self-Replicating Neural Programs*, 2021 (arXiv:2109.12786): replication of a training program in a toy language.
- BitNet b1.58 (Ma et al., 2024): ternary-weight language models, the reason k = 1 is a sane choice.
- Allen-Zhu & Li, *Physics of Language Models 3.3* (2024): ~2 bits of knowledge per parameter, the capacity bound that makes self-printing non-trivial.
