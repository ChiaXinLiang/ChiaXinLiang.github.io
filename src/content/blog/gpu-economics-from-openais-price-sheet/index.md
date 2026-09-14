---
title: 'Reading GPU Economics Off OpenAI''s Price Sheet'
description: "OpenAI's API prices are a compressed datasheet: every ratio on the page maps to a specific bottleneck in the silicon serving your tokens."
updatedDate: 'Sep 12 2026'
pubDate: 'Sep 13 2026'
heroImage: './section-overview.png'
code: 'econ-2'
order: 8
series: "efficient-ai"
level: intermediate
topic: "Economics"
tags: [economics, inference, pricing]
---

## Overview

![Concept overview: Reading GPU Economics Off OpenAI's Price Sheet](./section-overview.png)

50 to 1. That is the spread inside a single row of OpenAI's current price sheet. Output tokens from gpt-5.6-sol cost $20 per million, fresh input tokens cost $4, and cached input tokens cost $0.40. Nobody sat in a meeting and invented those gaps for marketing reasons. Each 1 is a hardware bottleneck with a dollar sign attached. Once you know what to look for, a public pricing page reads like a leaked engineering document.

This article decodes the page ratio by ratio. The prices themselves are public facts, and I pulled them from OpenAI's pricing page in September 2026. The mapping from price to silicon is inference on my part, so treat the ratios as approximate signals rather than an audited bill of materials. The signals, though, are remarkably consistent.

## Deep dive

### The vocabulary: 2 phases and a cache

![Deep dive: The vocabulary: 2 phases and a cache](./deep-dive-component-03.png)

Serving a large language model has 2 phases with opposite personalities.

**Prefill** is what happens to your prompt. The model reads all input tokens at once, in parallel, building up an internal record of what it has seen. This phase is *compute-bound*. The GPU performs enormous matrix multiplications, its tensor cores (the units that do bulk matrix math) run near their rated speed, and the whole prompt is digested in one pass over the model's weights.

**Decode** is what happens to the answer. The model produces output 1 token at a time, because each new token depends on the ones before it. For every single token, the GPU must stream the model's weights out of memory again. This phase is *bandwidth-bound*. The limit is not how fast the chip can multiply. It is how fast HBM, the stacked high-bandwidth memory soldered next to the GPU die, can feed it.

The bridge between the phases is the **KV cache**. During prefill, the model saves a compact summary of every input token. That summary holds the "keys" and "values" that attention uses to look back at earlier context. Decode reads this cache constantly. Send the same long prompt 2 times and the provider can save the KV cache from the first request. On the second it skips prefill entirely. That trick is what "cached input" pricing sells.

With those 3 ideas, the whole price sheet opens up.

### 4 prices, 4 bottlenecks

Here is the standard-tier row for gpt-5.6-sol, OpenAI's mid-flagship, as of September 2026, in dollars per million tokens:

| Line item | Price | Ratio to fresh input |
|---|---|---|
| Cached input | $0.40 | 0.1x |
| Fresh input | $4.00 | 1x |
| Cache write | $5.00 | 1.25x |
| Output | $20.00 | 5x |


Every model on the page shows the same shape. Cached input is exactly 10% of fresh input across the lineup, from the $10 gpt-6-astra down to the $0.20 gpt-5.6-luna. Cache writes carry a 1.25x premium everywhere. Output runs 5x input on the flagships and 6x on the smaller tiers. Long-context requests pay roughly 2x on input and 1.5x on output. Batch processing is half price. The low-latency "fast mode" tier is double. When 1 shape repeats across a dozen models at wildly different absolute prices, you are looking at cost structure, not positioning.

Let's decode each ratio.

**Cached input at 0.1x: reuse beats recompute.** On a cache hit, the provider skips prefill compute entirely. The KV summary of your prompt already sits in a memory tier somewhere. Serving your request means retrieving those bytes instead of re-deriving them with trillions of multiply-adds. A 90% discount says recompute is expensive and retrieval is cheap. The silicon says the same thing, with an interesting wrinkle we will get to below.

**Cache write at 1.25x: storage is not free.** The first time a prompt is cached, you pay full prefill *plus* a 25% surcharge. The surcharge covers moving the KV bytes into a persistent tier and holding them there against future hits. Modern serving stacks like Mooncake and LMCache spill KV state from scarce HBM into cluster DRAM and even SSDs. That pipeline has real costs: bandwidth to move the data, capacity to keep it, machinery to find it again.

**Output at 5x: decode is slow per token.** This is the roofline speaking. It deserves its own section below.

**Long context at ~2x: attention grows and the cache balloons.** Attention cost during prefill grows quadratically with sequence length. The KV cache grows linearly, hogging memory that would otherwise hold other users' requests. More on this below too.

### A worked example: 1 agent session, by hand

![Deep dive: A worked example: 1 agent session, by hand](./deep-dive-component-01.png)

Abstract ratios stick better with a concrete bill. Take a coding agent with a 50,000-token context (system prompt plus a repository digest) that runs for 20 turns. Each turn the user adds 1,000 tokens and the model replies with 500. The full history is resent every turn. All prices are gpt-5.6-sol standard tier.

**Without caching.** Turn *n* sends 50,000 + (n-1) x 1,500 input tokens. Over 20 turns:

- Input: 20 x 50,000 + 1,500 x (0+1+...+19) = 1,000,000 + 285,000 = **1,285,000 tokens**. At $4 per million: **$5.14**.
- Output: 20 x 500 = 10,000 tokens. At $20 per million: **$0.20**.
- Session total: **$5.34**, of which 96% is input.

**With caching.** Only the *new* tokens each turn are fresh; everything already seen is a cache hit. Roughly 78,500 tokens get written to cache over the session (the initial 50,000 plus 1,500 per subsequent turn), and the remaining ~1,206,500 input tokens are hits.

- Cache writes: 78,500 x $5 per million = **$0.40**.
- Cached reads: 1,206,500 x $0.40 per million = **$0.48**.
- Output: unchanged, **$0.20**.
- Session total: **$1.08**.

The bill dropped 5x, and its composition flipped. Output went from a rounding error, 4% of spend, to nearly a fifth of it. This is the general pattern for agentic workloads, which is why every serious agent framework became obsessed with prompt-cache hygiene. It also explains a breakeven rule you can derive from the sheet. The write premium is $1 per million tokens, the extra 0.25 x $4, and each later hit saves $3.60 per million, which is $4.00 minus $0.40. Caching pays for itself if a prefix has even a ~28% chance of being reused once. Almost any multi-turn conversation clears that bar on turn 2.

### Derive a cache decision without guessing provider costs

Let fresh input price be $$p_f$$, cache-write price $$p_w$$, and cache-hit price $$p_h$$, all in dollars per million tokens. If a written prefix receives an expected $$q$$ later billed hits, its incremental saving over fresh processing is positive when

$$
q(p_f-p_h)>p_w-p_f.
$$

For the listed prices of 4, 5, and 0.40 dollars, break-even expected reuse exceeds 0.2778 hits. This is an expectation across prefixes, not permission to charge a fraction of an actual request. Expiry, partial hits, and invalidated prefixes reduce realized reuse.

Compared with treating cached input as simply discounted input, this model includes the initial write premium. It supports a customer-side experiment: stabilize the prefix, record billed hit tokens, and compare complete session charges. It cannot recover the provider's actual cost from retail prices. Hardware efficiency, competitive positioning, demand, margins, and tier policy can all affect those prices. Roofline examples explain plausible engineering pressures. Neither the fivefold output ratio nor the cache discount uniquely identifies a deployed chip or a cost decomposition.

### Going deeper: the ratios, derived from the chip

![Deep dive: Going deeper: the ratios, derived from the chip](./deep-dive-component-02.png)

Now push 1 level down and ask why the ratios take these particular values. Use a concrete stand-in: a 70B-parameter dense model in FP8, 1 byte per weight, so 70 GB of weights. Run it on a GB300-class GPU with 288 GB of HBM at 8 TB/s, capable of very roughly 5 x 10^15 FLOPs of dense FP8 matrix math per second.

**Why output can cost more than input.** Generating 1 token requires about 2 FLOPs per parameter, so ~140 GFLOPs. At 5 PFLOP/s, that is 28 microseconds of arithmetic. But the GPU must also stream all 70 GB of weights through its compute units for that step. At 8 TB/s that takes 8,750 microseconds. For a single sequence, the chip spends over 99% of each decode step waiting on memory. In roofline terms, decode at batch size 1 has an arithmetic intensity of about 2 FLOPs per byte moved. The machine's balance point sits around 600 FLOPs per byte. Prefill, by contrast, amortizes 1 weight pass across thousands of prompt tokens and lands comfortably on the compute side of the roofline.


Providers claw back efficiency by batching many users' decode steps together, so 1 weight pass serves dozens of tokens. But batching has a ceiling, KV cache capacity, and a latency cost. Even well-batched decode produces tokens far slower than prefill consumes them. Public serving benchmarks give a feel for the gap. SGLang on a GB200 NVL72 rack reports roughly 26,000 prefill tokens per second per GPU against roughly 13,000 decode tokens per second per GPU on DeepSeek-R1-class models. That decode figure already assumes aggressive batching and disaggregated serving. Fold in the stricter latency guarantees on output and the 5-6x price multiple looks consistent with different engineering and service costs. It still does not reveal the provider's margin.

**Why cached input is 10x cheaper, not 1,000x.** Here is the wrinkle. Work out the raw silicon ratio yourself. Recomputing 1 prompt token costs ~140 GFLOPs, or 28 microseconds of GPU math in our stand-in. Now take the KV cache for that token in a GQA model with 80 layers and 8 KV heads of dimension 128 in FP8. It comes to about 2 x 80 x 1,024 bytes, roughly 160 KB. Reading 160 KB at 8 TB/s takes 0.02 microseconds. Recompute is on the order of a *thousand times* more expensive than the read, yet the discount is only 10x. The gap between 1,000x and 10x is everything that surrounds the read: keeping terabytes of KV state warm across DRAM and SSD tiers, shipping it back into HBM on a hit, indexing and evicting it, and eating the cost of misses. Caching at scale requires storage-system work. The retail discount alone does not quantify that overhead. It is not a free lunch. The DistServe retrospective's framing is apt: inference has quietly become a storage problem.

**Why long context costs ~2x.** At 160 KB per token, an 8K-token conversation carries about 1.3 GB of KV state. A 128K-token 1 carries about 20 GB. On our 288 GB GPU, after 70 GB of weights, the leftover memory fits roughly 160 short-context sequences but only about 10 long-context ones. Fewer concurrent sequences means each expensive weight read is shared fewer ways. Cost per token rises even before you count prefill's quadratic attention bill. OpenAI prices this as roughly 2x on input and 1.5x on output past the short-context threshold. The industry's answer at the hardware level is telling. NVIDIA's Rubin CPX is a GPU built specifically for long-context prefill, pairing heavy compute with cheaper GDDR7 memory because prefill does not need HBM's bandwidth the way decode does.


**Why batch is half price and fast mode is double.** These 2 tiers price the same thing in opposite directions: scheduling freedom. Batch jobs, with results within 24 hours, let the provider fill idle capacity and run at maximum utilization. They cost 50% of standard. Fast mode sells a different latency policy at 2x. Reserved capacity and scheduling can contribute, but the public price does not disclose occupancy or guarantee 0 queueing. Same silicon, different goodput contract.

### Common misconceptions

**"Cache discounts are a loyalty perk, like a bulk coupon."** No. The discount maps to compute the provider genuinely does not perform. DeepSeek gave an example consistent with efficiency affecting prices. On the day it shipped its sparse-attention architecture (DSA) in V3.2-Exp, it cut API prices by more than half and published the kernels. The concurrent announcement does not isolate costs, margins, or competitive pricing decisions.

**"Output tokens cost more because answers are worth more than prompts."** Value-based pricing would vary wildly by vendor and use case. Many products price output above input, and the roofline offers a plausible engineering explanation. Decode re-reads weights per token and stalls on memory bandwidth, while prefill amortizes 1 read across the whole prompt. Different workloads, products, and margin policies can produce different ratios. Retail convergence cannot uniquely identify hardware costs.

**"A cache hit costs the provider basically nothing, so 10% is a rip-off."** The raw compute-versus-read ratio is indeed closer to 1,000x than 10x. But the priced product is not a memory read. It is a distributed storage tier holding your KV state, about 160 KB per token and gigabytes per long conversation, across HBM, DRAM, and SSD. Transfer, indexing, eviction, and miss costs are all baked in. 10 percent of fresh price for all of that is closer to fair than it first appears. The 1.25x write premium is the honest admission that persistence costs money up front.

### The bigger picture

Once you read 1 price sheet this way, the whole market becomes legible. The 20x input-price spread between gpt-5.6-sol ($4) and gpt-5.6-luna ($0.20) tells you the frontier is no longer 1 flagship model but a routing ladder that customers can use for suitable workloads. The sheet does not disclose actual traffic shares. The long-context premium tells you memory capacity, not FLOPs, is the scarce resource, the same conclusion the [Blackwell-to-Rubin memory math](/blog/blackwell-to-rubin-memory-math/) reaches from the hardware side, where capacity stays flat at 288 GB while bandwidth nearly triples. The batch and fast-mode tiers are [goodput versus utilization](/blog/goodput-vs-utilization/) translated into retail pricing: you are buying a latency distribution, not just tokens. And the entire cached-input economy exists because of the KV cache, a data structure whose origin story is the attention mechanism itself, covered in [Attention in Plain Words](/blog/attention-in-plain-words/).

This is also, quietly, a recruiting pitch. Several tiers on that page correspond to useful optimization questions. Shifting any of them, whether a better cache hit rate, a leaner KV format, or a smarter batch scheduler, moves real revenue. That is precisely the job described in [What Does an ML Performance Engineer Do?](/blog/what-does-an-ml-performance-engineer-do/), except now the performance report is published monthly, in dollars, for everyone to read.

## Conclusion

- API price sheets suggest workload optimization opportunities: cached input at 0.1x prices skipped prefill compute, the 1.25x cache-write premium prices KV storage, output at 5-6x prices bandwidth-bound decode, and long-context premiums price the KV cache crowding out batch size.
- The roofline explains possible cost pressures, while prices also reflect product policy and competition. DeepSeek announced sparse attention alongside a price reduction, without a controlled decomposition of the causes.
- For anyone building on these APIs, the sheet is an optimization guide: maximize cache hits (stable prefixes, append-only context), budget output tokens hardest, and treat long context as a 2x luxury rather than a default.

### Sources

- OpenAI, "API Pricing," developer documentation: https://developers.openai.com/api/docs/pricing (prices retrieved September 2026)
- DeepSeek, "DeepSeek-V3.2-Exp Release" (DSA sparse attention with same-day 50%+ API price cut): https://api-docs.deepseek.com/news/news250929/
- Hao AI Lab, "DistServe: 18 Months Later," retrospective on prefill/decode disaggregation and KV-cache storage tiers: https://haoailab.com/blogs/distserve-retro/
- LMSYS, "Deploying DeepSeek on GB200 NVL72, Part 2," SGLang prefill/decode throughput measurements: https://lmsys.org/blog/2025-09-25-gb200-part-2/
- NVIDIA, "NVIDIA Unveils Rubin CPX," a GPU class dedicated to long-context prefill: https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference
- Mooncake: KV-cache-centric disaggregated serving (Qin et al., FAST '25 best paper), cited without link.

*Part of the [Efficient AI & Co-Design](/series/efficient-ai/) learning path. Browse its published articles by topic.*
