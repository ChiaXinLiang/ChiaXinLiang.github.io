---
title: 'Case File: Same Model, 3x Slower on the "Bigger" GPU'
description: "A team migrates from A100 to a newer, higher-TFLOPS instance and decode gets 3x slower; one spec-sheet line explains the whole regression."
pubDate: 'Sep 12 2026'
heroImage: './cover.png'
code: 'case-4'
order: 36
series: 'ai-performance'
topic: 'Troubleshooting'
tags: [troubleshooting, bandwidth, inference]
---

312 versus 362 TFLOPS. 80 GB versus 96 GB. One generation newer. The new instance beat the old one on every line quoted in the migration ticket, and decode throughput still fell from 52 tokens per second to 18. Nobody changed the model, the serving stack, or a single flag.

This is the fourth case file in the troubleshooting series, and it is the most common one I see in the wild, because it is baked into how cloud instances are marketed. A team serving a 13B model on a single A100 80GB gets a cost-optimization nudge: a newer instance type with two NVIDIA L40S GPUs is cheaper per hour, has 96 GB of total VRAM instead of 80, supports FP8, and carries an Ada Lovelace headline of "1,466 TFLOPS." They migrate. Time-to-first-token barely moves. Per-token decode speed drops by roughly 3x, users notice streaming has turned to molasses, and the rollback discussion starts before lunch.

The regression is not a bug. It is arithmetic, and you can predict it to within about ten percent from two datasheet numbers before you ever launch the instance.

## Decode is a memory race, not a math race

If the prefill/decode split is fuzzy, start with [how an LLM generates text](/blog/how-an-llm-generates-text/) and [TTFT and TPOT](/blog/ttft-and-tpot/); here is the one-paragraph version that matters for this case.

During decode, the model produces one token at a time per sequence. To produce that token, the GPU must read essentially every weight in the model from GPU memory, plus the accumulated KV cache, and it performs only about two floating-point operations per weight read. Two FLOPs per parameter, where each FP16 parameter is two bytes: that is an arithmetic intensity of roughly one FLOP per byte. An L40S can perform about 419 FLOPs in the time it takes to deliver one byte from its memory (362 TFLOPS dense FP16 against 864 GB/s). Batch-1 decode therefore sits about 400x below the ridge of the roofline. The tensor cores spend the overwhelming majority of every decode step waiting.

In that regime, the speed limit is simple:

```
time per token ≈ bytes touched per token / effective memory bandwidth
```

Bytes touched per token is model weights plus KV cache read. Effective bandwidth is the datasheet number times whatever fraction the kernels actually achieve, usually called MBU, model bandwidth utilization. FLOPS does not appear in the equation at all. Neither does VRAM capacity. Capacity decides whether the model *fits* and how much batch and context headroom you have; it says nothing about how fast bytes move.

That is the whole trap, and the spec sheets are built to spring it.

## The spec-sheet trap

Put the two cards side by side and read the lines in the order a procurement doc reads them:

| Spec line | A100 80GB SXM | L40S |
|---|---|---|
| Architecture | Ampere (2020) | Ada Lovelace (2023) |
| FP16 Tensor Core, dense | 312 TFLOPS | 362 TFLOPS |
| FP8 Tensor Core, dense | not supported | 733 TFLOPS |
| VRAM | 80 GB HBM2e | 48 GB GDDR6 |
| **Memory bandwidth** | **2,039 GB/s** | **864 GB/s** |

(Both columns are NVIDIA's own datasheet peaks, so treat them as theoretical ceilings, not achievable numbers.)

The L40S wins the architecture line, the FP16 line, and the FP8 line outright, and two of them beat one A100 on total VRAM. The only line it loses is the one that governs decode, and it loses it by 2.36x. The L40S is a superb card for what it was built for: graphics, video, and compute-dense inference such as diffusion models or high-batch prefill. It pairs Ada's big tensor throughput with GDDR6, which is far cheaper than HBM per gigabyte precisely because it moves far fewer bytes per second. (The [DRAM-to-HBM article](/blog/from-dram-to-hbm/) covers why that gap exists physically.)

![Newer silicon, less memory bandwidth: A100 80GB SXM — 312 TFLOPS dense FP16 — 2,039 GB/s HBM2e; L40S — ONE GPU — 362 TFLOPS dense FP16 — 864 GB/s GDDR6; WORKLOAD DECIDES — Prefill can benefit from more FLOPs — Batch-1 decode needs bandwidth. Data: NVIDIA A100 and L40S specifications](./spec-sheet-trap.png)

A useful habit: when someone says "bigger GPU," ask *bigger at what?* Every GPU is a point in a three-dimensional space of FLOPS, bandwidth, and capacity, and workloads project onto different axes. Batch-1 decode projects almost entirely onto bandwidth.

## The worked example: predicting the regression by hand

Take the actual workload: a 13B-parameter dense model served in FP16, typical chat traffic around 4,096 tokens of context, latency-sensitive so effective batch per GPU stays small. Call it batch 1 for clean arithmetic.

**Bytes per decode step.** Weights: 13B parameters at 2 bytes each is 26 GB, read once per token. KV cache: a 13B-class model has 40 layers and hidden size 5,120, so each cached token costs 2 (K and V) x 40 layers x 5,120 x 2 bytes ≈ 0.8 MB. At 4,096 tokens of context that is about 3.4 GB of cache to read per step. Total: roughly **29.4 GB per token**.

**A100 80GB.** Datasheet bandwidth 2,039 GB/s; a well-tuned serving stack on HBM typically achieves 70 to 80 percent MBU on this pattern. Take 75 percent: 1,529 GB/s effective.

```
29.4 GB / 1,529 GB/s ≈ 19.2 ms per token  →  ~52 tokens/s
```

**One L40S.** Datasheet 864 GB/s; GDDR6 tends to sustain a somewhat lower fraction on these access patterns, say 65 percent: 562 GB/s effective.

```
29.4 GB / 562 GB/s ≈ 52.3 ms per token  →  ~19 tokens/s
```

Predicted ratio: 2.7x. Observed in the ticket: 52 tokens/s down to 18, a 2.9x regression, with the last few percent coming from scheduler and launch overhead that a 52 ms step hides less well than you would hope. Two datasheet numbers and one MBU estimate reproduce the entire incident.

![Predicting the decode regression: SAME BYTE BILL — 26 GB weights + 3.4 GB KV — 29.4 GB moved per decode token; A100 — 75% MBU — 29.4 / (2,039 × 0.75) = 19.2 ms — ≈ 52 tokens per second; L40S — 65% MBU — 29.4 / (864 × 0.65) = 52.3 ms — ≈ 19 tokens per second. Illustrative model · MBU assumptions differ by GPU](./decode-prediction.png)

Notice what the calculation never asked for: TFLOPS, VRAM size, architecture generation, CUDA version. For a compute check, those 26 GFLOPs per token would take the L40S about 0.07 ms at datasheet FP16 throughput. The memory traffic takes 52 ms. During decode this "1,466 TFLOPS" card runs its tensor cores at well under one percent duty cycle.

## Going deeper: why the second GPU doesn't rescue you

The instance has two L40S cards, so the obvious counter is tensor parallelism: split the model across both, halve the bytes each card reads per token. Here the arithmetic gets quietly brutal.

First, aggregate bandwidth. Two L40S give you 2 x 864 = 1,728 GB/s of combined memory bandwidth. That is still **15 percent less than the single A100 you left**, before any parallelization overhead. There is no configuration of this instance whose total byte-moving capacity matches the old one.

Second, the interconnect tax. L40S has no NVLink; the two cards talk over PCIe Gen4 x16, about 32 GB/s per direction, substantially less than an NVLink-connected pair. Megatron-style tensor parallelism needs two all-reduces per transformer layer per token. Forty layers means 80 all-reduces per decode step, and at batch 1 each message is tiny (hidden size 5,120 in FP16 is 10 KB), so they are latency-bound rather than bandwidth-bound: each one costs tens of microseconds of PCIe round trip plus kernel launch and synchronization. Call it 30 to 50 µs each; that is another 2.5 to 4 ms per token that exists on neither card's datasheet.

Run the TP=2 numbers with the same 65 percent MBU: 29.4 GB across 1,728 GB/s effective-peak gives about 26 ms of memory time, plus ~3 ms of all-reduce, near 29 ms per token, or about 34 tokens/s. Better than one L40S, still 35 percent below the A100 baseline, and nowhere near the "two GPUs, twice the performance" mental model. Many teams skip TP entirely and run two independent replicas for throughput, which is often the right call, but then every individual user's stream runs at single-card speed: 19 tokens/s.

![Two cards add communication overhead: HBM/GDDR BANDWIDTH — Two L40S: 1,728 GB/s combined — One A100: 2,039 GB/s; TENSOR PARALLELISM — 80 small all-reduces per step — Assume ≈ 3 ms additional latency; PREDICTED TP2 STEP — ≈ 26 ms memory + 3 ms collectives — ≈ 34 tokens/s; baseline ≈ 52. Illustrative latency model · NVIDIA specifications](./tp2-pcie.png)

What *would* make the L40S instance competitive is cutting bytes, not adding cards. Quantize weights to FP8 (which the L40S executes natively) and the 26 GB read becomes 13 GB; W4A16 pushes it near 6.5 GB. With an FP8 KV cache as well, per-token traffic drops toward 9 GB and one L40S predicts around 60 tokens/s, beating the FP16 A100 setup. This is why L40S serving deployments are almost always quantized, and why "can we quantize?" is the first question to ask before this migration, not after. The trade-offs live in [quantization: what you gain, what you lose](/blog/quantization-what-you-gain-what-you-lose/).

## Common misconceptions

**"It has more VRAM, so it's faster."** VRAM is capacity, not speed. Capacity determines whether the weights fit and how much KV cache you can hold, which caps batch size and context length. It contributes nothing to how quickly a resident byte reaches the compute units. The 96 GB bought headroom this latency-bound workload never used, while giving up the 2 TB/s that it used on every single token.

**"FLOPS is the number that matters for AI workloads."** Only on the compute-bound side of the roofline. Prefill, with thousands of tokens processed in parallel, genuinely exploits big tensor throughput, which is why TTFT survived this migration almost untouched and why prefill-optimized parts like Rubin CPX exist. Batch-1 decode has an arithmetic intensity near 1 FLOP per byte against a machine balance of ~419; the 733 FP8 TFLOPS are physically unreachable in that phase. A spec line you cannot feed is a spec line you do not own.

**"Two GPUs are faster than one."** Not when each has 42 percent of the bandwidth and they share a PCIe bus. Aggregate bandwidth here is lower than the single card being replaced, and tensor parallelism at small batch adds dozens of latency-bound all-reduces per token. Multi-GPU is a tool for fitting bigger models and scaling throughput, and it earns its keep there; it is not a general speed multiplier, and over PCIe it can lose to one well-chosen card.

## The bigger picture

Underneath this case sits the single most useful reflex in performance work: classify the workload as compute-bound or memory-bound *before* comparing hardware, because the answer decides which spec line is load-bearing. That reflex is the subject of [compute-bound vs. memory-bound](/blog/compute-bound-vs-memory-bound/), and it is the same logic that separates latency machines from throughput machines in [CPU vs. GPU](/blog/cpu-vs-gpu-latency-vs-throughput-machines/). Vendors will always print the largest number on the box; your job is knowing which number your workload actually purchases.

It is also a neat portrait of [what an ML performance engineer actually does](/blog/what-does-an-ml-performance-engineer-do/). The engineer who caught this did no profiling and wrote no code. They read two datasheet lines, did thirty seconds of division, predicted 19 tokens per second, and matched the incident graph. The fix was a procurement decision, not a software one: either go back to bandwidth-heavy cards, or commit to quantization and make the cheap-per-hour instance genuinely cheap per token.

Napkin first, dashboard second. The napkin knew before the migration did.

## Takeaway

- Decode speed is bytes-per-token divided by effective memory bandwidth; TFLOPS and VRAM capacity are not in the formula. An L40S at 864 GB/s cannot out-decode an A100 at 2,039 GB/s in the same precision, no matter what the FLOPS lines say.
- Predict before you migrate: weights plus KV bytes, times a realistic 60 to 80 percent MBU, gives you tokens per second to within about ten percent. If the prediction says slower, the benchmark will too.
- Adding cards adds bandwidth only if the aggregate actually exceeds what you had, minus the interconnect tax; over PCIe at small batch that tax is steep. Cutting bytes through quantization is usually the stronger move on bandwidth-poor cards.

## Sources

- NVIDIA, A100 80GB Tensor Core GPU datasheet and product page (vendor peak specs): https://www.nvidia.com/en-us/data-center/a100/
- NVIDIA, L40S GPU product page (vendor peak specs): https://www.nvidia.com/en-us/data-center/l40s/
- Pope et al., "Efficiently Scaling Transformer Inference" (arXiv:2211.05102): https://arxiv.org/abs/2211.05102
- NVIDIA Developer Blog, "Mastering LLM Techniques: Inference Optimization": https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
- Williams, Waterman, and Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," Communications of the ACM, 2009.

*Part of the **AI Performance Engineering** series — previously: [Compute-Bound vs. Memory-Bound](/blog/compute-bound-vs-memory-bound/), the reflex this case file puts to work; next in the case files: quantized kernels that refuse to exist for your GPU.*
