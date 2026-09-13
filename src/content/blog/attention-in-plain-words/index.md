---
title: 'Attention in Plain Words: Every Token Looks at Every Other Token'
description: "The mechanism inside every modern LLM is a lookup that's softly blurred: each word asks the whole sentence what's relevant, and blends the answers. No relay, no fading memory."
pubDate: 'Sep 12 2026'
heroImage: './cover.png'
code: 'tf-1'
series: 'llm-basics'
topic: 'Transformer'
tags: ['attention', 'transformer']
---

"The animal didn't cross the street because **it** was too tired." What does *it* refer to?

You resolved that instantly — *it* means the animal, because "tired" fits animals, not streets. Swap "tired" for "too wide" and *it* flips to the street. Whatever machinery lets a model make that call is the heart of language understanding, and since 2017 that machinery has one name: **attention**. This article explains it with no equations.

## The problem attention solves

[Last article](/blog/rnn-lstm-and-the-wall/) ended with recurrent networks dying of two flaws: information faded as it was relayed word-by-word, and the relay forbade parallelism. The wish list for a successor was explicit — let every word connect to every other word *directly*, and let all of it happen *at once*.

Attention is exactly that: a direct, all-pairs connection.

![Attention resolving "it": the word directly consults every other word in the sentence and finds "animal" and "tired" most relevant — no relay through a running memory](./attention-lines.png)

## The mechanism: a soft lookup

Here is the whole idea in one metaphor. For each word, the model computes three things — think of them as three roles the word can play:

- a **query**: what am I looking for? (*it* is looking for: a thing that could be tired)
- a **key**: what do I offer as a match? (*animal* offers: I'm a living thing)
- a **value**: what information do I carry if you pick me? (the actual meaning-content of *animal*)

Each word's query is compared against every word's key, producing a **relevance score** for every pair. The scores are normalized into weights that sum to 1, and each word's new representation is the **weighted average of all the values** — mostly *animal*'s content with a dash of everything else, in our example.

![The soft lookup: one word's query is scored against every word's key; the scores weight a blend of all the values — retrieval, softly blurred](./soft-lookup.png)

That's it. Attention is a lookup table where, instead of retrieving one entry, you retrieve *all* entries blended in proportion to relevance. Three details worth appending:

- **All the queries, keys, and values are produced by weights** — learned by [the same gradient descent as ever](/blog/how-models-learn/). Nobody tells the model that "tired" relates to "animal"; that emerges from predicting text.
- **It runs several times in parallel** ("multi-head" attention): one head might track pronoun reference, another syntax, another nearby words. Each head is the same mechanism with its own learned weights.
- **Distance doesn't exist.** Word 1 attends to word 1,000 exactly as easily as to word 2. The fading-relay problem is simply gone — which is why long documents became feasible.

## A worked example you can follow by hand

Abstract mechanisms stick better with numbers, so let's run a miniature attention step. Take the three-word input "cat sat down" and pretend each word's query and key are just 2-number vectors:

| word | key vector | value (informal) |
|---|---|---|
| cat | (1.0, 0.2) | "furry agent" |
| sat | (0.1, 0.9) | "past action" |
| down | (0.2, 0.8) | "direction" |

Suppose *sat* is computing its new representation, and its **query** is (0.3, 1.0) — informally, "I'm a verb; who's my subject, and what modifies me?" Score each word by the dot product (multiply matching positions, add up):

- vs *cat*: 0.3×1.0 + 1.0×0.2 = **0.50**
- vs *sat*: 0.3×0.1 + 1.0×0.9 = **0.93**
- vs *down*: 0.3×0.2 + 1.0×0.8 = **0.86**

Normalize those into weights that sum to 1 (the real model uses softmax, which also sharpens the winners) — roughly 0.24 / 0.41 / 0.35. *Sat*'s updated representation becomes 0.24×(cat's value) + 0.41×(its own) + 0.35×(down's): still mostly "a past action," now measurably flavored with *who* did it and *which way*. Every word in the sentence does this simultaneously; that's one attention layer. Real models do it with 128-number vectors and dozens of heads, but the arithmetic you just did is the whole mechanism.

The engineering aside worth planting now: notice each word needed its key and value available for everyone else's lookup. During generation, models **cache** those keys and values instead of recomputing them per token — that's the KV cache whose memory appetite drives half the serving economics in [the performance series](/blog/goodput-vs-utilization/).

## Multi-head: several lenses at once

A single attention pattern is one "lens" on the sentence. Real blocks run 8–128 heads in parallel, each with its own learned query/key/value weights, each free to specialize. Interpretability work on real models has found heads that track subject-verb agreement, heads that link closing brackets to opening ones, heads that follow coreference chains like our *it*→*animal* example, and many that defy tidy description. The outputs of all heads are concatenated and mixed — so each token's update draws on many relationship types simultaneously.

Why not one big head with more capacity? Because ten cheap specialists beat one expensive generalist here: different linguistic relationships want *differently shaped* similarity comparisons, and separate heads let each comparison be learned independently. It's the same "give the architecture the right structure" lesson as [CNN filters](/blog/cnn-how-machines-learned-to-see/) — many small pattern-matchers, reused everywhere.

## Common misconceptions

**"Attention is what the model 'focuses on,' like human attention."** The name invites the analogy, but resist it: attention weights are just learned similarity scores that route information. High weight on a word doesn't mean the model "cares about" it in any human sense, and researchers have shown attention maps can be misleading as explanations of *why* a model answered as it did.

**"Each word attends to a few relevant words."** No — every token attends to *every* token, always. The weights are merely concentrated on a few. The compute cost is paid for all pairs regardless of how peaked the distribution is; that's exactly why the quadratic cost is unavoidable in vanilla attention.

**"Attention replaced neural networks."** Attention layers are *made of* the [same weighted sums](/blog/what-is-a-neural-network/) as everything else — the queries, keys, and values are produced by ordinary learned matrices, and attention alternates with plain feed-forward layers in the full architecture ([next article](/blog/transformer-architecture-in-one-picture/)). It's a new wiring diagram, not new physics.

## Why this won: it fits the hardware

Notice what the mechanism *doesn't* have: any dependence between positions during the computation. Every word's lookup can happen **simultaneously** — the whole thing is a few large matrix multiplications, which is precisely the operation GPUs are built to do in bulk.

This is the architecture-meets-hardware moment this series keeps circling. [CNNs](/blog/cnn-how-machines-learned-to-see/) encoded "images are local and repetitive." Attention encodes "any word may relate to any word" — and, crucially, does so in a *parallel-friendly* form. The 2017 paper that proposed building models from attention alone was titled, with earned confidence, ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

One honest cost, which becomes a running theme in the performance series: all-pairs comparison means the work grows with the *square* of the sequence length. Double the document, quadruple the attention compute. Much of modern LLM engineering — from FlashAttention to sparse attention — is the industry negotiating with that square. ([The KV cache](/blog/goodput-vs-utilization/), a serving-side consequence, gets its own article later.)

## Takeaway

- Attention = every token directly scores its relevance to every other token, then takes a weighted average of their content. A lookup, softly blurred.
- Query/key/value are all learned; multiple heads run the mechanism in parallel with different learned specialties. Distance costs nothing — long-range understanding stops being special.
- It won because it fits both the data ("anything can relate to anything") and the hardware (all positions compute at once). The price: compute grows with the square of sequence length.

## Sources

- Vaswani et al. (2017). ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- Alammar — ["The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/) (the canonical visual walkthrough; the "it" example follows its presentation)
- Olah & Carter (2016). ["Attention and Augmented Recurrent Neural Networks"](https://distill.pub/2016/augmented-rnns/), *Distill*

---

*Part of the **Fundamental of LLM** series. Previous: [RNN and LSTM](/blog/rnn-lstm-and-the-wall/). Next: the full Transformer architecture in one picture.*
