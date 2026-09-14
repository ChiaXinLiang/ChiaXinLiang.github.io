---
title: 'The 9.3% Model: Qwen3-Next Reports 32B Quality at Lower Training Cost'
description: "Qwen3-Next pairs linear attention with extreme MoE sparsity to hit 32B-class quality on 9.3% of the reported training cost — here is the arithmetic behind both levers."
pubDate: 'Sep 13 2026'
updatedDate: 'Sep 12 2026'
heroImage: './section-overview.png'
code: 'cd-1'
order: 9
series: "efficient-ai"
level: advanced
topic: "Co-Design Cases"
tags: [moe, attention, efficiency]
---

## Overview

![Concept overview: The 9.3% Model: Qwen3-Next Reports 32B Quality at Lower Training Cost](./section-overview.png)

9.3%. That is the share of Qwen3-32B's training cost in GPU-hours the Qwen team says it took to train Qwen3-Next-80B-A3B, a model they report as matching or beating the dense 32B model across their benchmark suite. Under 10% of the reported GPU-hours, for the same class of quality — with more than 10x the inference throughput once the context grows past 32K tokens.

Those are vendor numbers, published on Qwen's own blog, and we should treat them that way: self-reported, on benchmarks the vendor selected. But the claim is not magic, and that is exactly why it is worth unpacking. The reported efficiency reflects architectural and training choices, including 2 prominent mechanisms that anyone can reason about with pencil and paper. This article does that arithmetic.

## Deep dive

### Where the compute actually goes

![Deep dive: Where the compute actually goes](./deep-dive-component-03.png)

Start with what a training bill is made of. For a transformer trained on D tokens with N parameters touched per token, the standard approximation for training compute is:

$$
C \approx 6ND.
$$

The 6 covers 1 forward pass (2 FLOPs per parameter per token, 1 multiply and 1 add) plus the backward pass (roughly 2 times the forward). This approximation, popularized by the OpenAI scaling-laws work, ignores attention's context-dependent term, which is fine at short contexts and increasingly not fine at long ones. So a model's compute bill has 2 knobs:

1. **How many parameters does each token actually touch?** In a dense model, every token flows through every weight. N equals total parameter count, full stop.
2. **How much does each token pay to look at the tokens before it?** In standard attention, token number 50,000 compares itself against all 49,999 predecessors. That cost is invisible in the 6ND formula but grows quadratically with context length, and it can become substantial at long contexts.

Qwen3-Next attacks both knobs at once. That is the whole trick.

**Lever 1: sparse activation.** Qwen3-Next is a mixture-of-experts (MoE) model with 80 billion total parameters, of which only about 3 billion activate for any given token. An MoE layer replaces the transformer's single feed-forward block with many parallel "expert" blocks and a small router network that picks a few experts per token. The model's total capacity — the amount of knowledge it can store — scales with all 80B parameters. The compute per token scales only with the ~3B that fire. Capacity and compute, which a dense model welds together, come apart.

**Lever 2: hybrid linear attention.** 3 out of every 4 layers replace standard attention with Gated DeltaNet, a linear-attention variant. Instead of storing every past token's key and value and comparing each new token against all of them, a linear-attention layer maintains a fixed-size state — think of it as a running summary matrix — that gets updated once per token. Cost per token: constant, regardless of whether the context holds 1K tokens or 256K. The remaining 1 layer in 4 keeps standard (gated) attention, for reasons we will get to, because pure linear attention has a known weakness.


Neither lever is new on its own. MoE dates to the 1990s and returned at scale with Google's sparsely-gated LSTM work in 2017. Linear attention has a 5-year paper trail. What Qwen3-Next demonstrates is that stacking both, aggressively, holds up at frontier quality — and that the savings can combine, subject to memory and routing overhead.

### The worked example: an equal-token FLOP comparison

![Deep dive: The worked example: an equal-token FLOP comparison](./deep-dive-component-01.png)

Take the 6ND approximation and compare the 2 models on the same token budget D.

**Dense Qwen3-32B.** Every token touches all 32B parameters:

$$
C_{\mathrm{dense}} \approx 6\times32\times10^9 D.
$$

**Qwen3-Next-80B-A3B.** Every token touches only the ~3B active parameters (the router, shared expert, attention, and embeddings are all inside that figure):

$$
C_{\mathrm{next}} \approx 6\times3\times10^9 D.
$$

Divide:

$$
\frac{C_{\mathrm{next}}}{C_{\mathrm{dense}}}\approx\frac3{32}=0.09375.
$$

Qwen reports 9.3% of training cost in GPU-hours. The equal-token FLOP estimate happens to be close, but these are different quantities: their numerical proximity does not verify the reported cost. The useful MoE intuition is: **the training bill tracks active parameters, not total parameters.** An 80B-total model with 3B active costs roughly what a 3B dense model costs to push a token through. The other 77B parameters sit in memory, waiting for the router to call on them, contributing capacity but not FLOPs.


2 honest caveats. First, this is an equal-token hypothetical. Qwen states that Next used 15T tokens sampled from Qwen3's 36T-token corpus and reports cost in GPU-hours, not a full audited FLOP accounting. Different token budgets, attention work, and achieved hardware efficiency change the comparison. Second, matching quality while reporting lower GPU-hours is the empirical claim doing the heavy lifting. Scaling folklore long held that sparse models need far more total parameters to match dense quality (Qwen3-Next uses 2.5x) and that training them stably at high sparsity is hard. The benchmark suite tests the quality result; the division illustrates sparse activation under an equal-token assumption.

Now the second lever, which the 6ND formula hides. At 32K context and beyond, standard attention's quadratic term stops being a rounding error. For inference the pain shows up as the KV cache: a standard attention layer must keep keys and values for every past token. Try illustrative numbers: a 48-layer model storing 2 KB of KV per token per layer (in FP16) needs 96 KB per token across a fully standard stack, which at a 128K context is about 12.6 GB per sequence, before you serve a second user. Cut standard attention to 12 of 48 layers and the cache drops to about 3.1 GB; the 36 DeltaNet layers hold small fixed-size states whose memory does not grow with context at all. That is where the reported >10x long-context prefill and decode throughput comes from, and it compounds with the MoE savings: fewer FLOPs per token, and each token drags far less memory traffic behind it. If you want the mechanical intuition for why the KV cache exists in the first place, [Attention in Plain Words](/blog/attention-in-plain-words/) builds it from scratch.

### Going deeper: why keep any full attention at all?

![Deep dive: Going deeper: why keep any full attention at all?](./deep-dive-component-02.png)

If linear attention is so cheap, why not use it everywhere? Because the fixed-size state is a lossy summary. Standard attention retains an explicit route to token 17 from token 50,000, though successful verbatim recall is learned rather than guaranteed, which is what you need for copying a serial number out of a document, matching a bracket 40K tokens back, or needle-in-a-haystack recall. Linear attention compresses history into a state matrix of constant size, and information theory is unforgiving about what a constant-size state can hold from an unbounded stream. Pure linear-attention models score well on perplexity and fall over on exact-recall tasks.

The hybrid is therefore not a compromise; it is a division of labor. DeltaNet layers handle the bulk of sequence mixing cheaply. The periodic full-attention layers act as precise random-access memory over the context. Qwen's ablation-driven 3:1 ratio says you need less exact retrieval than a uniform stack provides, but you cannot drop it to 0. IBM reached a similar design point independently with Granite 4.0, interleaving Mamba-2 and transformer layers 9:1 and claiming (their own charts) over 70% serving-RAM reduction at long context. Different linear-time operator, same architectural verdict.

The "gated" in Gated DeltaNet matters too. The delta rule updates the state selectively, overwriting the slot associated with the current key rather than adding on top of everything, and the gate lets the model decay stale state. Both mechanisms fight the mushing-together that plagued earlier linear attention. This is the same lineage of ideas we saw when [RNNs and LSTMs hit their wall](/blog/rnn-lstm-and-the-wall/): gates deciding what to keep and what to forget, now rebuilt inside a parallel-trainable operator.

There is a stability story underneath as well, and it is easy to miss. High-sparsity MoE training historically suffered from router collapse and loss spikes, each spike costing a rollback and wasted GPU-days. Qwen3-Next ships a bundle of unglamorous fixes (0-centered layernorm weights and careful router normalization among them) aimed precisely at spike-free training. Moonshot's Kimi K2 makes the same point at larger scale: 1 trillion total parameters, 32B active — a 3.2% activation ratio — trained on 15.5 trillion tokens with, per the technical report, 0 loss spikes, credited largely to the MuonClip optimizer. Stability is an efficiency feature. A run that never restarts is a run whose every FLOP counts, which is [goodput by another name](/blog/goodput-vs-utilization/).


Zoom out across 2025-26 open releases and the direction is unmistakable. DeepSeek-V3 activates 5.5% of its parameters, OpenAI's gpt-oss-120b about 4.4%, Qwen3-Next 3.8%, Kimi K2 3.2%. Each generation lights up a smaller fraction of a larger whole, and quality keeps climbing anyway.

A gated delta update makes the fixed-state mechanism more precise. 1 simplified single-head convention first decays state S and then applies a correction for the current key:

$$
\widetilde S_t=\alpha_t S_{t-1},\qquad
S_t=\widetilde S_t+\beta_t(v_t-\widetilde S_t k_t)k_t^\top,\qquad
o_t=S_t q_t.
$$

Here S maps keys to values, k, v, and q are current vectors, alpha is a decay gate, and beta a write-strength gate. Normalization and implementation ordering vary, so this illustrates the delta-rule mechanism rather than reproducing every Qwen kernel detail. With scalar state 2, alpha equal to 0.5, beta equal to 0.25, key 1, and value 3, the decayed state is 1 and the corrected state is 1.5. The update moves the stored prediction toward the new value without retaining a separate entry for every previous token.

This bounded state changes historical storage compared with full KV attention, but it can lose distinctions that explicit retained keys preserve. Periodic full-attention layers add another retrieval path. Test exact recall and long-context quality as well as throughput. The 3/32 parameter ratio predicts 9.375% only under equal token budgets and the simplified 6ND accounting; matching Qwen's reported 9.3% closely does not constitute an independent derivation of the real training bill.

### Common misconceptions

**"3B active means it's basically a 3B model."** No. Per-token compute matches a 3B dense model; capacity does not. Different tokens route to different experts, so across a document the model draws on far more than 3B distinct parameters. Benchmarks bear this out: 3B-class dense models do not touch Qwen3-32B, while Qwen3-Next reportedly does. The correct mental model is a library with 80B books and a rule that any single question may consult only 3B of them. The library's usefulness is set by the collection, not the per-question quota.

**"Linear attention has replaced full attention."** It has not, and Qwen3-Next is itself the counterexample: the team kept full attention in 25% of layers after ablations, precisely because pure linear stacks fail at exact recall. Every credible production hybrid (Qwen3-Next, Granite 4.0, the various Jamba-family models) retains some full-attention or full-KV layers. The honest headline is that full attention became a seasoning rather than the base, not that it disappeared.

**"9.3% training cost means ~10x cheaper for users, verified."** 2 errors in 1 sentence. Training savings and inference savings are different quantities; the inference win here comes mostly from the KV-cache and throughput side, is context-length dependent, and total-parameter memory (80B weights must live somewhere, [and memory is the scarce resource](/blog/blackwell-to-rubin-memory-math/)) partially offsets it. And "verified" overstates things: the 9.3% figure, the benchmark parity, and the >10x throughput are all Qwen's own measurements. They are plausible and consistent with the arithmetic above, corroborated in direction by independent labs, but no neutral party has audited them.

### What this does to the scaling narrative

For a few years the field's working assumption was that capability tracked training FLOPs along smooth power laws, so the leaderboard belonged to whoever bought the most compute. Qwen3-Next and its cohort complicate that story in a specific way: these results show that architecture and training design can shift empirical efficiency; they do not establish a universal scaling law. A 10x improvement in capability-per-FLOP is worth the same, on the leaderboard, as a 10x bigger cluster, and 2025-26 delivered it through model design rather than procurement: extreme sparsity, hybrid attention, and (per Kimi K2) optimizers that eliminate wasted restarts. DeepSeek's V3.2-Exp closed the economic loop publicly by shipping sparse attention and a same-day 50%+ API price cut.

This matters most because the industry is running into a wall that money cannot quickly fix: power. With grid interconnect queues running 5 to 7 years, the marginal frontier model increasingly gets trained inside a fixed energy envelope, and capability-per-FLOP becomes capability-per-megawatt. Architecture is the one lever that improves it without waiting on a substation. It also feeds back into hardware: sparse, hybrid models change what chips should be good at (memory capacity for resident experts, less HBM bandwidth burned on KV traffic), which is exactly the co-design loop this series keeps circling. The [transformer's original all-attention design](/blog/transformer-architecture-in-one-picture/) was itself a bet on what 2017 hardware did well; Qwen3-Next is the same bet re-placed for 2026.

The 9.3% model is not the end state. It is 1 clean, public data point on a curve the whole field is now racing down: how little of a model can you light up, and how little of the past can you store exactly, before quality notices?

## Conclusion

- Training compute tracks **active** parameters, not total: 6ND with N = 3B versus 32B predicts 9.375% at equal token budgets. Qwen's 9.3% measures reported GPU-hour cost, which this division cannot independently establish.
- The 2 levers multiply: MoE sparsity cuts FLOPs per token, hybrid linear attention cuts the context-dependent cost and shrinks the KV cache roughly 4x at a 3:1 ratio, driving the reported >10x long-context throughput.
- Every headline number here is vendor-reported; the trend, though, is corroborated across independent labs (Kimi K2 at 3.2% activation, Granite 4.0's 9:1 hybrid, DeepSeek's DSA price cut), and it points at capability-per-FLOP as the metric that now moves the frontier.

### Sources

- Qwen team, "Qwen3-Next: Towards Ultimate Training & Inference Efficiency" — [qwen.ai blog](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)
- Kimi Team, "Kimi K2: Open Agentic Intelligence" — [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)
- DeepSeek, "DeepSeek-V3.2-Exp release: sparse attention and API price reduction" — [api-docs.deepseek.com](https://api-docs.deepseek.com/news/news250929/)
- IBM, "Granite 4.0: hyper-efficient, high-performance hybrid models" — [ibm.com](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)
- Kaplan et al., "Scaling Laws for Neural Language Models" — [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Yang, Kautz, Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule" (ICLR 2025)

*Part of the [Efficient AI & Co-Design](/series/efficient-ai/) learning path. Browse its published articles by topic.*
