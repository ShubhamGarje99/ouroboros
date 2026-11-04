# Ouroboros: A Language Model That Prints Its Own Weights

*Shubham Garje, COEP Technological University, Pune. Draft, November 2026.*

## Abstract

We train a small transformer language model whose parameters are small integers
and whose vocabulary includes tokens for those integers and for addresses of its
own weight rows. Prompted with an address, the model emits the integers stored at
that address. Printing every address and loading the result into an empty copy of
the architecture gives back the same model. The self-printing objective is trained
jointly with next-character prediction, and its target is the model's own current
weights, so training chases a fixed point in which the weights both model text and
describe themselves. We report how close to that fixed point training gets for
ternary and 7-level weights, what it costs in language-model loss, how the copy
degrades or stabilises over generations, and what happens when the model prints
itself with sampling noise, which turns self-printing into reproduction with
mutation and lets a population of copies improve without gradients. To our
knowledge this is the first transformer, and the first language model, that
reproduces its own parameters; the only prior neural-network quine (Chang & Lipson,
2018) was a feed-forward regressor that approximated its weights.

## 1. Setup

**One table.** The model is a character-level transformer (d = 64, 2 layers, 4 heads,
context 128) whose every parameter lives in one ternary table T of shape rows × 64.
The first rows are the embeddings of the 65 text characters, the three value
tokens (−1, 0, +1), three control tokens (`<Q>`, `<END>`, `<SCALE>`), 64 scale
tokens and 64 coordinate tokens; then 128 position rows; then, per layer, the rows
of W_q, W_k, W_v, W_o (64 each), W_1 (256), the gate G (256) and W_2ᵀ (256). That is
2,375 rows, 152,000 ternary weights, and 2,375 tokens: every row is a token and
every token is a row. The output head is tied to the table. Weights are the codes
times one global scale that is snapped to a 64-level log grid and printed by
`<Q> <SCALE>`.

**Block.** Pre-norm attention; the MLP runs in parallel on the same normalised
input and is gated by the attention contribution:
  a = W_o · Attn(n(x)),  h = GELU(W_1 n(x)) ⊙ (1 + G a),  x ← x + a + W_2 h.
With G = 0 this is an ordinary parallel transformer block. The gate is what makes
"coordinate c of a copied row" a near-linear circuit.

**Self-printing.** The prompt `<Q> <row r> <col c>` is answered with the value
token for T[r][c]. The loss is a cross-entropy over the three value tokens only,
exactly the set the printer decodes among, mixed with the next-character loss on
Shakespeare. The target is the model's current quantised table. A full self-print
is 152,000 three-token prompts, batched, a few seconds on a CPU.

**Why the weights must be the vocabulary.** A network with weight entropy H bits
can hold at most H bits, and a memorised description of its own weights costs
exactly H bits, for any quantisation and any sparsity. So no model can *memorise*
itself and still do anything else. A program quine escapes the same bound
because its interpreter supplies a quote operation (a string literal). Here the
embedding lookup is that operation: because weight row r is also token r, the
model can place its own row into the residual stream and copy from it. The
self-printing circuit then costs a fixed number of weights (identities and three
constants) regardless of model size, and the rest of the table is free for the
language-model task.

**Constructive check.** The circuit can be written down by hand, and doing so is
what turned up the two architectural requirements. Attention is *strictly causal*
(a position never attends to itself); otherwise every row whose content resembles
the attention markers (the `<Q>` row, the key-reader rows, the marker position row)
makes position 2 attend to itself instead of to the row. Seven coordinates of the
64 are reserved: `<Q>` = e_Q0 and every head's keys read −x[Q0] in four
dimensions, so at the answer position the row (key ≈ 0) always beats `<Q>` (key −32);
position 2's embedding is e_J + e_J2 and queries read x[J]; every ordinary text
position carries +e_Y + e_Y2, which every identity unit subtracts, so the whole
circuit is silent on text; value tokens embed as ±e_A and +e_Z. W_v = W_o = I copies
the normalised row; three identity units per coordinate compute
GELU(x[c] − x[Y] − x[Y2]) · (1 + a[c]); the four marker coordinates are printed by
difference units (x[J] − x[J2] and the like, which ±1 weights can express);
and the unit counts (3, 5, 4, 3 and the two offset units) are chosen so that, after
RMSNorm, the offset equals the "value 0" level exactly for every kind of prompt. Each
rule here was forced by a measured failure of the previous version, and every
coordinate, including the markers themselves, is printable, which is what makes the
copy of a copy identical. Installed into a fresh table at scale 1.5 with no training,
the circuit prints **152,000 of 152,000 weights** and the scale token; the printout
loaded into an empty model is bit-identical to the printer, and so is its printout,
for as many generations as we ran.

## 2. Why self-printing is a capacity problem

A network with N quantised parameters, each taking one of 2k+1 values, has a
description of N·log2(2k+1) bits if the values are unstructured. Its ability to
store arbitrary facts is bounded by a few bits per parameter: about 2 bits per
parameter for knowledge learned from text (Allen-Zhu & Li, 2024) and about 3.6
bits per parameter for raw memorisation of random data (Morris et al., 2025).
With k = 3 the description costs 2.8 bits per weight, so a 7-level network can
only print itself if its weights are far from random. With ternary weights and
80% zeros, the description costs about 0.9 bits per weight (the entropy of the
code distribution), and self-printing fits inside the memorisation budget with
room left over for the language-model task.

Two mechanisms push the weights toward self-consistency during training. The
first is the self-printing loss itself, whose target is the current quantised
weights. The second is a *bidirectional* form of that loss: the target is a
soft distribution over levels derived from the shadow weight's distance to the
rounding boundaries, and gradients flow into the shadow weight, so a weight that
the model confidently "remembers" as zero is pulled toward zero. A third,
discrete mechanism, regeneration, overwrites a weight with the model's own
prediction when the model is confident and disagrees, following Chang & Lipson.

Experimentally, the moving target is not the bottleneck: freezing a snapshot of
the weights and training only the self-printing loss against it memorises just as
slowly (31% of a 7-level snapshot after 2,000 steps) as the live target. The
bottleneck is the number of passes over the rows: memorising ~1,600 rows of
near-random symbols is a many-epoch problem.

## 3. Experiments

All runs: 2,375 rows × 64 = 152,000 ternary weights, 2 layers, 4 heads, context
128, character-level Shakespeare (1.1 MB, 90/10 split), CPU only (M3, ~11 steps/s).

### 3.1 An exact fixed point, then a language model on top of it

The constructed circuit prints 152,000/152,000 weights with no training. Training
the language model on top of it, with the circuit's own entries and the reserved
coordinates of every row held fixed (the free rows are the 65 character
embeddings, 125 position rows, the second layer and the unused hidden units of the
first), reaches a validation loss of 2.95 nats/char in 6,000 steps while the
self-print stays exact at every checkpoint. `verify_tm` then prints the trained
model, loads the printout into an empty model, and finds it bit-identical,
scale included; that child's printout is bit-identical too, for the three
generations we ran. This is the first language model that reproduces its own
parameters exactly.

The language model is weak (2.95 nats/char; a plain 2-layer model of this size
reaches about 2.2 to 2.5), because the circuit occupies the first layer and seven
of the 64 coordinates, and because training at the weight scale the circuit needs
(1.5) is noisy. Random initialisation of the second layer, kept silent on the
self-print prompt, was necessary for the language model to train at all: from an
all-zero second layer, almost no code flips in 2,000 steps.

### 3.2 What training discovers on its own

Trained from random initialisation with the self-printing loss, the same
architecture finds a partial copy circuit within about 2,000 steps: 78 % of
weights with ternary codes, 80 % with binary codes, with language-model losses of
2.50 and 2.18. The error analysis is unambiguous: the learned circuit reads the
*sign* of a weight (88 % on −1, 91 % on +1) and is at chance on *zero* (27 %).
A sign is a linear readout of the copied row; zero needs a clean copy and a
constant offset, which is exactly the part of the hand-built circuit that took the
most care (unit counts matched to RMSNorm, strictly causal attention). Binary
weights remove the zero class and still plateau near 80 % because attention at
the small weight scales training prefers is not sharp (scores scale with the
square of the weight scale).

### 3.3 Memorising oneself does not work, and cannot

The first version of this project (separate matrices, two-digit row addresses, no
quote mechanism) never left the marginal rate across ternary and 7-level weights,
sparsity, soft targets and regeneration, and memorising a frozen snapshot of the
weights was just as slow. The counting argument in Section 2 says why.

### 3.4 Reproduction with mutation

Printing with sampling noise turns the printer into a reproduction operator with
a mutation rate set by the temperature. At τ = 1.5 about 16 % of the weights
mutate; at τ = 0.3 about 1 %. Both are lethal: the child's loss rises (2.95 → 6.1
at 1 %) and the child can no longer print itself, so the lineage collapses in a
single generation. The copier is part of what it copies and random mutation hits
it in proportion to its size; a viable operator would have to spare the circuit.
Reported as a demo, not a result.

## 4. Limitations

Rows are printed one address at a time, not as one unbroken sequence. The
architecture and tokenizer (the "interpreter") are not printed. Results are on a
125k-parameter character model; nothing here says what happens at scale.

## 5. Related work

Chang & Lipson 2018; Schmidgall 2021; BitNet b1.58 (Ma et al. 2024); Allen-Zhu & Li
2024 (capacity); Hofstadter's self-reference; von Neumann's self-reproducing
automata (the copier/description split that this model collapses into one object).
