---
title: 'Two series, one goal: understand what makes AI fast'
description: 'What this blog covers — LLM fundamentals for newcomers, and AI systems performance engineering for the people running the GPUs.'
pubDate: 'Sep 12 2026'
tags: ['meta']
---

A modern GPU cluster reporting 100% utilization can be wasting most of its compute. A "bigger" GPU can run the same model slower. And the difference between a profitable AI product and a money pit often comes down to engineering decisions nobody sees.

This blog documents what I'm learning about that invisible layer, in two parallel series:

**Fundamental of LLM** — for engineers getting into AI: how neural networks actually learn, why the Transformer won, and how an LLM turns your prompt into tokens. No prior ML background assumed.

**AI Performance Engineering** — for people who run models on real hardware: GPU memory math, CUDA kernels, the KV cache, batching and quantization, serving at scale, and the cluster infrastructure that keeps thousands of GPUs fed.

Every article is a standalone read: one concept, concrete numbers, original diagrams, primary sources cited.

New articles land twice a week. The short versions show up on [my LinkedIn](https://www.linkedin.com/) — the full write-ups live here.
