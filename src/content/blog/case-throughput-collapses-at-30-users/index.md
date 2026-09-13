---
title: 'Case File: Throughput Collapses at 30 Concurrent Users'
description: "A serving cluster that hums at 25 users falls off a cliff at 30 — the culprit is a 32 GB KV cache pool, and the fix is arithmetic, not hardware."
pubDate: 'Sep 12 2026'
updatedDate: 'Sep 12 2026'
heroImage: './cover.png'
code: 'case-2'
order: 18
series: "llm-serving"
level: advanced
topic: "Production Serving"
tags: [kv-cache, inference, troubleshooting]
---

At 25 concurrent users the deployment served 1,420 tokens per second. At 30 users it served 650. 20 percent more load, 54 percent less throughput, and p99 time-to-first-token went from 900 ms to 41 seconds. Nothing crashed. No OOM error, no restart, no alert beyond the latency dashboard turning red.

This illustrative reconstruction is the second case file in the troubleshooting series, and it is the most common production incident I know of in LLM serving. The failure mode has a specific shape: performance is not merely flat past some load level, it is dramatically *worse* than at lower load. A queue that saturates degrades gracefully. This system fell off a cliff. Cliffs mean the server is doing something pathological under pressure, and in this case the pathology has a name: KV cache preemption thrash.

## The symptom, precisely

The setup: a 70B-parameter model, weights quantized to 4-bit, serving on a single 80 GB GPU with a popular inference engine (the numbers below use vLLM's vocabulary, but SGLang and TensorRT-LLM have the same machinery under different names). Requests average about 4,000 tokens of context. Load testing showed beautiful scaling from 1 to 25 concurrent users: aggregate tokens per second climbed almost linearly, per-user latency crept up only mildly. Classic continuous-batching behavior, exactly what the [batching lever](/blog/batching-the-biggest-throughput-lever/) is supposed to buy you.

Then, between 27 and 30 users, 3 things happened at once:

1. Aggregate tokens/sec dropped by more than half.
2. p99 TTFT exploded from under a second to tens of seconds, while p50 stayed almost normal. (If TTFT and TPOT are new vocabulary, the [basics article](/blog/ttft-and-tpot/) covers them.)
3. GPU compute utilization *fell*. The SMs were less busy during the collapse than before it.

That third observation is the tell. If the GPU were compute-starved, utilization would be pinned at 100% and throughput would plateau. Utilization dropping while load rises means the GPU is spending its time on something other than useful token generation. The bottleneck is not FLOPs. It is memory capacity.

## What actually runs out

During decode, every live request holds its attention history in the KV cache: 1 key vector and 1 value vector per token, per layer. The engine pre-allocates a fixed pool for this at startup, which is why `nvidia-smi` shows ~78 GB used even with 2 users connected. PagedAttention manages that pool the way an OS manages RAM: it is carved into fixed-size blocks (16 tokens each by default), and sequences are given blocks on demand, so almost no memory is wasted on fragmentation. The paper that introduced this (Kwon et al., 2023) measured 60-80% of KV memory wasted in pre-PagedAttention engines; modern engines waste under 4%.

But paging only eliminates *waste*. It does not create capacity. The pool is a hard budget, and every admitted request draws from it continuously as it decodes, 1 block every 16 generated tokens. When the pool runs dry mid-generation, the scheduler faces a choice with no good options: some running sequence must give its blocks back.

vLLM's scheduler handles this by preempting a low-priority sequence, in one of 2 modes. **Swap** copies its blocks to CPU RAM over PCIe and copies them back later. **Recompute** simply drops the blocks and, when the sequence is rescheduled, re-runs its entire prefill from scratch. Recompute is the default for typical single-sequence requests because a fresh prefill is often faster than paging tens of gigabytes across PCIe, and recomputed prefill restores the corresponding KV state within the engine's numerical behavior, so nothing is lost except time.

Except time is exactly the thing being measured. A preempted 4,000-token request that gets recomputed costs the GPU a full second-scale prefill that produces 0 new output tokens. Do that continuously and the cliff appears.

![Admission crosses the memory limit: LOW CONCURRENCY — More sequences reuse each weight read — Aggregate throughput improves; KV CAPACITY LIMIT — Growing contexts fill the KV pool — The limit depends on live token count; PREEMPTION — Recompute competes with decode — Throughput can collapse. Original qualitative schematic · PagedAttention (2023)](./fig-cliff.png)

## The worked example: finding the cliff by hand

Let's predict the exact user count where this deployment runs out, using a Llama-3.3-70B-shaped model. Its attention geometry, from the Llama 3 technical report: 80 layers, 64 query heads, but only **8 KV heads** thanks to grouped-query attention (GQA), each with dimension 128.

**KV bytes per token** (FP16 cache):

```
2 (K and V) × 80 layers × 8 KV heads × 128 dims × 2 bytes
= 327,680 bytes ≈ 320 KiB per token
```

**Per request at 4,096 tokens of context:**

```
327,680 B × 4,096 ≈ 1.34 GB per request
```

**Now the pool.** The engine claims 90% of the 80 GB card, so 72 GB is the working budget. The 4-bit weights of a 70.6B model take about 37 GB once you include the higher-precision embeddings and quantization scales. Activations, CUDA graph buffers, and workspace eat roughly another 3 GB. What remains is the KV pool:

```
72 − 37 − 3 = 32 GB
32 GB ÷ 327,680 B/token ≈ 97,600 tokens of KV capacity
97,600 ÷ 4,096 tokens/request ≈ 23.8 requests
```

The pool holds about 23 fully-resident requests, a couple fewer once you account for contexts growing as decode proceeds. Below that, adding users makes throughput go *up*, because decode is memory-bandwidth-bound and bigger batches amortize the cost of streaming 37 GB of weights per step. At 24-26 users the pool is essentially full and the scheduler is squeezing new arrivals into blocks freed by finishing requests. At 30 users there is structurally no room: 5 or 6 requests are perpetually preempted, recomputed, and preempted again. The load test found the cliff at "around 30." The equal-length arithmetic says 24. That is a capacity warning, not an exact prediction of a 30-user measured threshold.

![An 80 GB GPU memory budget: FIXED COSTS — 37 GB weights + 3 GB activations — 8 GB reserved for runtime/headroom; KV POOL — 80 − 37 − 3 − 8 = 32 GB — 1.34 GB per representative request; ADMISSION BOUND — floor(32 / 1.34) = 23 requests — The next request must wait. Illustrative budget · Llama 3 architecture; decimal GB](./fig-memory.png)

2 counterfactuals make the geometry vivid. If this model used old-style multi-head attention with 64 KV heads, KV would cost 2.5 MiB per token, each request would need 10.7 GB, and the same pool would hold **2** users. GQA's 8× reduction is the only reason 23 fit at all, which is why you should never size a deployment from parameter count alone; the KV-head count is a first-class input. Conversely, quantize the cache itself to FP8 and the per-token cost halves to 160 KiB, moving the cliff from ~24 users to ~47.


Admission should budget blocks and future growth rather than an average user count. Let $$k$$ be KV bytes per token, $$q$$ tokens per block, $$s_i$$ currently cached tokens for request $$i$$, and $$g_i$$ its reserved generation allowance. A conservative capacity test is

$$
kq\sum_i\left\lceil\frac{s_i+g_i}{q}\right\rceil\le M_{\mathrm{pool}}.
$$

Assume no shared prefixes and a uniform uncompressed cache layout. For $$k=327680$$ bytes and $$q=16$$, 1 block costs 5,242,880 bytes. A request with 4096 cached tokens and 512 reserved output tokens needs 288 blocks, or 1.50994944 decimal GB. A 32 GB pool can admit at most 21 such requests, leaving approximately 0.291 GB; a 22nd needs approximately 33.219 GB.

This improves the baseline policy by anticipating cache growth before admission instead of discarding completed prefill after exhausting capacity. The reservation can be intentionally conservative; smaller allowances improve occupancy but require explicit queuing or preemption policy when requests exceed them. Shared prefixes can reduce physical allocations, but must be counted through actual ownership and reference tracking. The 24-request arithmetic bound and a stipulated 30-user collapse are not an exact match: variable lengths and scheduler behavior can explain a range, but require measurements. Treat this case as an illustrative reconstruction. Recomputed values should preserve model semantics within expected numerical tolerance; bit-identical results are not guaranteed across kernel schedules.


## Going deeper: why a cliff and not a slope

Queueing systems normally degrade gracefully: past saturation, throughput holds at capacity and waiting time grows. The KV-exhausted system does something worse because preemption-by-recompute makes the server *destroy completed work* under overload.

Trace 1 cycle of the loop. The pool is full and a running sequence gets preempted; its 256 blocks (4,096 tokens ÷ 16 tokens/block) are freed and immediately claimed by other sequences' decode growth. When the victim is rescheduled, the engine must re-run its full prefill: roughly 10^12-scale FLOPs that were already paid for once. That prefill occupies the GPU for time in which the whole batch generates nothing, so decode throughput dips. It also refills 1.34 GB of pool, which pushes the pool back over the edge and selects the next victim. The system converges to a steady state where a meaningful fraction of all compute is re-prefill, which is precisely why SM utilization can look "busy" while tokens/sec craters: the GPU is working hard producing KV entries it already produced before.

This is also why p99 diverges while p50 barely moves. Preemption victims are not chosen uniformly; the scheduler evicts from the back of its priority order, so the same unlucky requests get recomputed repeatedly while fresh short requests sail through. A request preempted 3 times pays 4 prefills. Your p99 *is* that request. Median-only dashboards hide the entire incident, the same trap covered in [Tokens per Second: What It Hides](/blog/tokens-per-second-what-it-hides/).

The observability fix is knowing which counters tell the truth. `nvidia-smi` memory usage is useless here (it reads ~78 GB at every load level, because the pool is pre-allocated). The counters that matter in vLLM: `vllm:num_preemptions_total` (any sustained nonzero rate is this incident), `vllm:gpu_cache_usage_perc` (pinned at ~100% during the cliff), and the scheduler's running-vs-waiting queue depths. The engine even logs a warning the first time it preempts, citing reduced performance. In my experience that log line is the single highest-value grep in LLM serving.

## The fixes, in the order I'd try them

**1. Quantize the KV cache to FP8.** 1 flag in vLLM (`kv_cache_dtype="fp8"`) and supported natively in TensorRT-LLM. Halves per-token cost, roughly doubles the user count at the cliff, and measured quality deltas on modern models are small (validate on your own evals; vendor accuracy claims are self-reported). This is the highest leverage-to-effort ratio available.

**2. Admission control that respects the arithmetic.** Cap concurrently scheduled sequences (`max_num_seqs`) near the computed capacity, ~22 here, so request 25 waits in queue instead of triggering thrash. Counterintuitively, admitting fewer requests raises delivered throughput, because queued requests cost nothing while preempted ones burn prefill. This is a pure goodput move in the sense of the [goodput article](/blog/goodput-vs-utilization/): utilization looks lower, useful work goes up. Set `max_model_len` honestly too; capacity math done at "mean context" fails when a 30k-token request lands.

**3. Size for the real context distribution and GQA geometry.** The formula above takes 5 minutes per model. Run it before procurement, not after the incident. A second GPU with tensor parallelism doesn't just double FLOPs, it roughly quadruples this KV pool (weights split across cards, so each card's leftover grows).

**4. Tiered offload for the cache.** Swap-to-CPU preemption, or a proper KV tier like LMCache and vLLM's CPU-offload connector, turns eviction from "destroy and recompute" into "demote and reload." PCIe reload of a 1.34 GB context takes tens of milliseconds; the recomputed prefill takes ~1 s of GPU time. Offload shines when contexts recur (multi-turn chat), where it composes with prefix caching.

**5. Route prefill elsewhere.** At larger scale, the reason this incident happens at all is that prefill and decode fight for 1 pool. [Disaggregating them](/blog/the-prefill-decode-disaggregation-story/) gives decode nodes a KV budget that prefill bursts can't invade.

![The preemption feedback loop: POOL FULL — Scheduler preempts a sequence — Its discarded KV must be rebuilt; RECOMPUTE — Prefill consumes compute and time — Active decodes make less progress; BREAK THE LOOP — Admit by KV token budget — Use validated KV quantization. Original scheduler schematic · vLLM preemption docs](./fig-thrash.png)

## Common misconceptions

**"Throughput collapsed, so we need more compute."** No. Compute-bound saturation produces a plateau at 100% utilization, not a collapse with falling utilization. This GPU had FLOPs to spare; it had no free KV blocks. Buying a faster card with the same 80 GB moves the cliff almost nowhere, whereas an FP8 cache flag doubles it for free. Diagnose the resource before spending on 1.

**"`nvidia-smi` shows 78/80 GB used, so we're already at the memory limit at 5 users."** The engine pre-allocates the KV pool at startup; near-full device memory is the *healthy* state and tells you nothing about load. The meaningful gauge is pool occupancy (`gpu_cache_usage_perc`) and the preemption counter. Teams routinely misread the pre-allocation as a leak and lower `gpu_memory_utilization` to "fix" it, which shrinks the pool and moves the cliff to fewer users.

**"Continuous batching means overload degrades gracefully."** Continuous batching degrades gracefully only while the pool has room. Past exhaustion, recompute-mode preemption makes marginal load *subtract* capacity, since every admitted-then-evicted request converts finished prefill work into future rework. Graceful degradation under overload is a property you must engineer with admission control; no scheduler gives it to you for free once memory runs out.

## The bigger picture

This case is the KV cache's revenge for being invisible. Weights are static and easy to budget; the cache is dynamic, proportional to live traffic, and the first thing to run out in production. That is why the industry's last 2 years of serving work is mostly KV-cache work: GQA and MLA shrink it at the architecture level, FP8 and paged layouts shrink it at the systems level, offload tiers and disaggregation give it a memory hierarchy of its own. If the mechanics of prefill versus decode underlying all of this are fuzzy, the [generation basics article](/blog/how-an-llm-generates-text/) is the foundation; the economic framing of why every one of these fixes is really a cost lever lives in the [goodput piece](/blog/goodput-vs-utilization/).

The meta-lesson for troubleshooting: a performance *cliff* is a fingerprint. Plateaus point at saturated compute or bandwidth. Cliffs point at a resource with hard admission semantics, where crossing the limit triggers expensive corrective machinery, page thrash in an OS, retry storms in an RPC mesh, preemption in an LLM scheduler. When you see 1, ask what the scheduler does when it runs out, not what it does when it's busy.

## Takeaway

- KV capacity, not compute, sets the concurrency ceiling: for a GQA 70B at 4k context on 1 80 GB card, ~320 KiB/token × 4,096 tokens ≈ 1.34 GB per request against a ~32 GB pool, so the cliff sits near 24 users, and the back-of-envelope predicts the load test.
- The collapse mechanism is preemption thrash: recompute-mode eviction re-runs whole prefills, so past exhaustion each extra user subtracts throughput and p99 TTFT diverges while p50 looks fine. Watch preemption counters and cache occupancy, not `nvidia-smi`.
- Cheapest fixes first: FP8 KV cache (~2× capacity), admission control at the computed limit (queue, don't thrash), then GQA-aware sizing, CPU offload tiers, and prefill/decode disaggregation as scale grows.

## Sources

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023) — https://arxiv.org/abs/2309.06180
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — https://arxiv.org/abs/2305.13245
- Grattafiori et al., *The Llama 3 Herd of Models* (model architecture: 80 layers, 8 KV heads) — https://arxiv.org/abs/2407.21783
- vLLM documentation and source (preemption, swap vs. recompute, `kv_cache_dtype`, metrics) — https://docs.vllm.ai and https://github.com/vllm-project/vllm
- NVIDIA TensorRT-LLM (FP8 KV cache support) — https://github.com/NVIDIA/TensorRT-LLM

*Part of the [LLM Inference & Serving](/series/llm-serving/) learning path. Browse its Beginner, Intermediate, and Advanced topics and planned articles.*
