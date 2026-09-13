---
title: 'The Transformer Architecture in One Picture'
description: "Attention plus a feed-forward layer, wrapped in residual connections, stacked N times. That's the entire blueprint behind GPT — here's the picture and the reason for every part."
pubDate: 'Sep 12 2026'
heroImage: './cover.png'
code: 'tf-2'
series: 'llm-basics'
topic: 'Transformer'
tags: ['transformer', 'architecture']
---

GPT-3 is 96 copies of the same block, stacked. Llama, Claude, Gemini, DeepSeek — different sizes, same block. Learn one diagram and you've read the blueprint of essentially every modern language model.

[Last article](/blog/attention-in-plain-words/) explained attention, the star mechanism. This one assembles the full machine around it — and explains *why* each supporting part exists, because every part earns its place.

## The block

![The Transformer block: attention (tokens talk to each other) then a feed-forward layer (each token thinks alone), each wrapped with residual connections and normalization — stack N times. Redrawn simplified from Vaswani et al., 2017](./transformer-block.png)

A Transformer is: turn words into number-vectors, push them through **N identical blocks**, and read a prediction off the end. Each block does just two operations, in order:

**1. Attention — the tokens talk to each other.** Every token gathers relevant context from every other token, as covered last time. After this step, *it* knows it means "the animal."

**2. Feed-forward — each token thinks alone.** A small two-layer network (the plain [multiply-add-squash kind](/blog/what-is-a-neural-network/)) processes each token *individually*, no cross-talk. If attention is the meeting, feed-forward is everyone going back to their desk to digest what they heard. Unglamorous, but it holds roughly two-thirds of the model's weights — much of an LLM's "knowledge" is widely believed to live here.

That alternation — gather context, process it, gather again with sharper questions, process again — repeated dozens of times, is the entire computational story. Early blocks resolve grammar and reference; later blocks handle increasingly abstract relationships. Votes about votes, one more time.

## The supporting cast (each solves a real failure)

Three more components appear in the diagram, and none is decoration:

- **Positional information.** Attention treats input as an unordered set — "dog bites man" and "man bites dog" would look identical. Fix: stamp each token's position into its vector before the blocks. (How you stamp it matters for long documents; modern models use rotary variants — a later series covers this.)
- **Residual connections.** Every block's output is *added to* its input rather than replacing it — each block writes edits onto a running document instead of rewriting from scratch. This is ResNet's skip-connection trick, and it's what lets gradients flow through 96 blocks without [vanishing](/blog/rnn-lstm-and-the-wall/). Without residuals, deep Transformers simply don't train.
- **Normalization.** Keeps each layer's numbers in a healthy range so training stays stable across dozens of blocks. Bookkeeping, but load-bearing bookkeeping.

## One blueprint, two famous variants

The 2017 original had two towers (an encoder reading the source sentence, a decoder writing the translation). The field then split it:

- **Encoder-only** (BERT-style): every token attends in both directions; great for *understanding* tasks like search and classification.
- **Decoder-only** (GPT-style): each token may only attend to tokens *before* it — because the training game is "predict the next word," and peeking ahead would be cheating. This is the variant that ate the world; when people say "LLM" today they almost always mean a decoder-only Transformer.

![Causal masking: in a decoder-only Transformer each token may attend only to earlier tokens — the rule that makes next-word training honest and the KV cache possible](./causal-mask.png)

That "only look backward" rule, called causal masking, has a huge practical consequence: past tokens' computations can be cached and reused while generating — the KV cache that dominates the serving economics covered in [the performance series](/blog/goodput-vs-utilization/).

## Takeaway

- The Transformer is one block — attention (tokens confer) + feed-forward (tokens digest), with residuals and normalization — stacked N times. GPT-3 is 96 of them.
- Every support part fixes a specific failure: positions restore word order, residuals let 96-deep gradients survive, normalization keeps training stable.
- Modern LLMs are decoder-only Transformers: attend-backward-only, which enables next-word training — and the KV caching that defines inference economics.

## Sources

- Vaswani et al. (2017). ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (the block figure is redrawn simplified from Figure 1)
- He et al. (2015). ["Deep Residual Learning"](https://arxiv.org/abs/1512.03385) (residual connections)
- Alammar — ["The Illustrated GPT-2"](https://jalammar.github.io/illustrated-gpt2/) (decoder-only walkthrough)

---

*Part of the **Fundamental of LLM** series. Previous: [Attention in plain words](/blog/attention-in-plain-words/). Next: why Transformers won — the parallelism story.*
