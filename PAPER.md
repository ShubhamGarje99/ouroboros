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

_(architecture, quantisation, token layout, losses; see README for now)_

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

### 3.1 How close to a fixed point
### 3.2 The cost in language-model loss
### 3.3 Generations: copy of a copy of a copy
### 3.4 Reproduction with mutation

## 4. Limitations

Rows are printed one address at a time, not as one unbroken sequence. The
architecture and tokenizer (the "interpreter") are not printed. Results are on a
125k-parameter character model; nothing here says what happens at scale.

## 5. Related work

Chang & Lipson 2018; Schmidgall 2021; BitNet b1.58 (Ma et al. 2024); Allen-Zhu & Li
2024 (capacity); Hofstadter's self-reference; von Neumann's self-reproducing
automata (the copier/description split that this model collapses into one object).
