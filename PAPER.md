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

_(N·log2(2k+1) bits of description vs ~2 bits/parameter of capacity; structure
as the way out; the self-printing loss as description-length pressure)_

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
