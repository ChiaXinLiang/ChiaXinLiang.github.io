---
title: 'Case File: The Quantized Model Isn''t Faster'
description: 'An illustrative INT4 deployment saves weight traffic but does not improve throughput. Analyze kernel support, overhead, and batching.'
pubDate: 'Sep 12 2026'
updatedDate: 'Sep 12 2026'
heroImage: './section-overview.png'
code: 'case-6'
order: 22
series: "llm-serving"
level: advanced
topic: "Inference Methods"
tags: [quantization, troubleshooting, gpu]
---

## Overview

![Concept overview: Case File: The Quantized Model Isn't Faster](./section-overview.png)

A 7B model in FP16 is 14 GB of weights. Quantized to INT4 with group scales it is about 3.9 GB. That is a 3.6x cut in the bytes the GPU must stream from HBM for every decode step. In this illustrative incident, the team measures throughput before and after: 41 tokens/sec per request, then 42. Someone re-ran the benchmark 3 times because the result looked like a mistake. It wasn't. The INT4 model was genuinely no faster. In 1 configuration it was 15% slower than the FP16 baseline it was supposed to beat.

This case shows up constantly, because the reasoning that leads to the deployment is correct as far as it goes. Decode is memory-bound. Weight bytes dominate memory traffic. Quarter the weight bytes, near-quadruple the decode speed. Every link in that chain is a real mechanism. I've written up the memory-bound argument in detail in [the quantization gains-and-losses article](/blog/quantization-what-you-gain-what-you-lose/). But the chain has 2 hidden preconditions: the kernels must actually exploit the small format, and weight bytes must actually be the thing dominating your traffic. This case file is about what happens when either one fails.

## Deep dive

### What weight-only INT4 actually promises

First, be precise about what was deployed. GPTQ and AWQ, the 2 dominant INT4 schemes for LLMs, are *weight-only* quantization: W4A16. Weights are stored as 4-bit integers plus FP16 scale factors, typically 1 scale per group of 128 weights. Additional format metadata and unquantized tensors explain a footprint near 3.9 GB rather than the 3.5-GB packed payload. Activations stay in FP16, and so does the arithmetic. Inside a proper W4A16 kernel, the weights are dequantized to FP16 on the fly. The multiply-accumulate then runs on the ordinary FP16 tensor core path.

That means W4A16 buys you exactly 1 thing: fewer bytes read from HBM per matrix multiply. It does not buy you faster math. There is no INT4-times-FP16 tensor core instruction. The compute cost is identical to the FP16 model, plus a small dequantization overhead. So the entire speedup lives or dies on whether memory traffic was your bottleneck. It also depends on whether the kernel that runs preserves the traffic reduction.

Hold that framing and the 3 root causes fall out almost mechanically.

### The investigation: 3 suspects

![Deep dive: The investigation: 3 suspects](./deep-dive-component-03.png)

**Suspect 1: the fallback dequant path.** The runtime loaded the INT4 checkpoint but had no fused kernel for this GPU, shape, or quant config, so it fell back to the naive implementation. That path launches a kernel to dequantize the INT4 weights into an FP16 buffer, then hands that buffer to a regular cuBLAS GEMM. Count the bytes. The dequant kernel reads 3.9 GB of packed weights and *writes 14 GB of FP16* to HBM. The GEMM then reads that 14 GB back. Total traffic per decode step: roughly 32 GB, versus 14 GB for the plain FP16 model. The quantized model now moves more than 2 times the data of the baseline. This is how INT4 ends up slower. It is exactly what the 15%-regression configuration was doing.


The fix is a fused kernel: Marlin (from IST Austria's DASLab, now the default GPTQ/AWQ path in vLLM on Ampere and newer), the AWQ GEMM kernels, or a TensorRT-LLM engine built with weight-only quantization enabled. These dequantize in registers and shared memory while streaming data toward the tensor cores. The FP16 version of the weights never touches HBM at all.

**Suspect 2: mixed paths that idle the tensor cores.** A subtler variant of the same disease. Some fallback W4A16 implementations do the dequant-plus-multiply on CUDA cores rather than tensor cores. Wiring an on-the-fly format conversion into the tensor core MMA pipeline is genuinely hard engineering. On an H100 the FP16 tensor core path is roughly 15x the FLOP rate of the FP32 CUDA core path, so a kernel that saves 3.6x on bytes while giving up an order of magnitude on math throughput is a terrible trade everywhere except the most extremely memory-bound corner. Nsight makes this visible: pipe utilization shows FMA pipes busy and tensor pipes near 0.

**Suspect 3: the workload was never weight-bound.** This is the one that gets experienced teams, because nothing is misconfigured. The serving stack runs continuous batching, and effective batch size sits at 200+ concurrent sequences. At that operating point the GEMMs may approach their delivered compute ceiling: the 14 GB of weights are read once per step but amortized over hundreds of tokens of arithmetic. Shrinking bytes that no longer set the critical path does nothing. The next section puts numbers on exactly where that flip happens.

### Worked example: finding the crossover batch

![Deep dive: Worked example: finding the crossover batch](./deep-dive-component-01.png)

Take the 7B model on an H100 SXM: 3.35 TB/s of HBM3 bandwidth and 989 TFLOPS of dense FP16 tensor core throughput. NVIDIA's spec sheet headline of 1,979 is the 2:4 sparsity number, and dense is half of it.

Each decode step, every weight participates once, so per step the GPU must:

- **Move weight bytes:** 14 GB in FP16, or 3.9 GB in INT4. At 3.35 TB/s that floor is **4.2 ms** (FP16) or **1.2 ms** (INT4), independent of batch size.
- **Do the math:** a forward pass is about 2 FLOPs per parameter per token, so 14 GFLOP per token, or 14·B GFLOP for a batch of B sequences. At the ideal 989 TFLOPS that costs **0.0142·B ms**, growing linearly with batch.

The step time is roughly the larger of the 2. Now solve for where compute catches the memory floor:

- FP16: 0.0142·B = 4.2 ms at **B ≈ 296**
- INT4: 0.0142·B = 1.2 ms at **B ≈ 82**


Read the chart in 3 zones. Below batch ~82, both formats are riding their memory floors and INT4 delivers nearly its full 3.5x. Between ~82 and ~296, INT4 has already hit the compute wall while FP16 is still memory-bound, so the gap shrinks continuously. At batch 150, FP16 takes 4.2 ms and INT4 takes 2.1 ms, and the "3.5x" quantization is now a 2x speedup. Above ~296, both formats take the same 0.0142·B ms and quantization buys 0 throughput.

Those crossovers are best-case, computed at peak FLOPS. Real decode GEMMs are skinny, batch-of-200 rows against 4,096-wide matrices, and they typically reach 40 to 60% of peak. That drags the crossovers down to roughly batch 120 to 180 for FP16 and 35 to 50 for INT4. A production vLLM deployment running continuous batching at effective batch 150 to 250 sits squarely in that zone. Weight-only INT4 was always going to return a fraction of its advertised gain there. Nobody did anything wrong. The bound moved.

There is a second ambush hiding at large batch: KV-cache reads. Every decode step also streams each sequence's cached keys and values. For a full-attention 7B model that is about 0.5 MB per cached token in FP16. At batch 64 with a mean context of 2,048 tokens, that is 64 × 2,048 × 0.5 MB ≈ 67 GB per step, dwarfing the 14 GB of weights entirely. Weight-only quantization does not touch a single 1 of those bytes. When KV traffic dominates, the levers are GQA, KV quantization, and paging, not weight formats.

The diagnosis needs a model containing all candidate bounds. Let B be effective batch size, P parameters, D_w weight traffic, D_kv cache traffic per iteration, beta delivered bandwidth, C delivered arithmetic throughput, and e conversion plus launch overhead:

$$
t_{\mathrm{step}}\gtrsim\max\left(\frac{D_w+D_{\mathrm{KV}}}{\beta},\frac{2PB}{C}\right)+e,\qquad
B^*\approx\frac{D_w C}{2P\beta}.
$$

The crossover B_star excludes cache traffic and assumes constant delivered rates. With P equal to 7 billion, D_w equal to 3.9 GB, C equal to 989 TFLOPS, and beta equal to 3.35 TB/s, it is 82.25. For 14-GB FP16 traffic it is 295.22. These are optimistic roofline crossings, not guarantees that batch 83 is compute-bound.

A group scale alone does not explain a 3.9-GB checkpoint: 4-bit payload plus 1 FP16 scale per 128 weights is approximately 3.609 GB for 7 billion weights. Extra 0 points, unquantized tensors, alignment, and the actual parameter count can increase it. Read the checkpoint layout instead of inferring metadata from its filename. Profile fused and fallback paths at matched batch and context, measure HBM bytes, and then test quality. A lower weight footprint can improve capacity even when conversion, compute, or cache traffic prevents a latency gain.

### Going deeper: what Marlin actually does

![Deep dive: Going deeper: what Marlin actually does](./deep-dive-component-02.png)

It's worth seeing why a fused W4A16 kernel is hard enough that fallbacks exist at all. Marlin (Frantar et al., 2024) is the reference design. It earns its near-ideal 3.87x speedup at batch 1 through a stack of mechanisms:

- **Offline weight reshuffling.** The INT4 weights are permuted once at load time into an interleaved layout matched to the tensor core fragment format. The kernel can then feed MMA instructions without shuffling data across threads at runtime.
- **Dequantization in registers.** Packed 4-bit values are expanded to FP16 using bit-twiddling tricks: masking nibbles into the mantissa field of a magic FP16 constant, then 1 subtraction. Those cost a handful of logic instructions instead of a lookup, which keeps the conversion off the critical path.
- **Software pipelining.** Asynchronous copies (`cp.async`) stage the next weight tiles from HBM into shared memory while the current tiles are being dequantized and multiplied. The tensor cores and the memory system therefore stay busy simultaneously.
- **Striped partitioning.** The work is split so all SMs stay loaded even for the skinny GEMMs of small-batch decode. A naive tiling would leave most of the chip idle.

The kernel sustains close to full memory bandwidth while the tensor cores run the FP16 math. That is the definition of winning in a memory-bound regime. The practical consequence for a troubleshooter: this machinery only engages when shapes, GPU architecture, group size, and activation ordering all match what the kernel supports. Miss 1 and the runtime silently picks the slow path.

Which is why the single highest-value diagnostic step in this case is reading kernel names in a profile. Run the server under Nsight Systems and look at what actually executes per decode step. Names containing `marlin` or `gptq_marlin_gemm` mean the fused path is live. A pair per layer, some `dequantize`-flavored kernel followed by a generic `s16816gemm` from cuBLAS, means you are on the fallback. Your HBM traffic went up, not down. Just 5 minutes of profiling replaces a week of speculating, a habit I've argued for since [the ML performance engineer job description](/blog/what-does-an-ml-performance-engineer-do/).

### Common misconceptions

**"INT4 means the GPU does 4-bit math."** For GPTQ/AWQ-style W4A16, no. Storage is 4-bit; arithmetic is FP16 after on-the-fly dequantization, at the same FLOP cost as the unquantized model. Formats where the multiply itself runs in 4-bit hardware (NVFP4 on Blackwell tensor cores) are a different design with different constraints, covered in [the 4-bit format war](/blog/nvfp4-vs-mxfp4-the-4bit-format-war/). Confusing the 2 leads to expecting compute speedups that W4A16 cannot deliver by construction.

**"Quantization always speeds up inference."** It speeds up the regimes where weight bytes set the critical path: small-batch decode, single-user latency, memory-capacity-constrained serving. Compute-bound regimes, meaning prefill at almost any size and decode at high batch, see roughly nothing. A fallback dequant path is actively slower than FP16, because it moves 2.3x the baseline's bytes. The honest claim is "quantization raises the memory-bound ceiling," which is only a speedup if that ceiling was the one pressing on you.

**"The checkpoint quantized cleanly, so deployment is done."** Perplexity numbers validate the *format*; they say nothing about the *kernels*. The identical GPTQ checkpoint can run 3x faster than FP16 or 15% slower, depending on whether the backend has a fused kernel for your exact GPU generation, group size, and quant config. Runtimes rarely fail loudly when they fall back. Quality evaluation and performance validation are separate sign-offs. The second one requires a profiler, not a benchmark harness average.

### The bigger picture

This case is the troubleshooting series' cleanest illustration of a rule that generalizes: **optimizations target a specific bound, and they only pay when that bound is yours.** Weight-only quantization attacks weight bytes. If your step time is set by compute, KV traffic, scheduling gaps, or network, INT4 weights are dead weight-savings. The same logic explains why prefill and decode want different hardware entirely (prefill is compute-hungry, which is [how Rubin CPX got its own chip](/blog/prefill-gets-its-own-chip-rubin-cpx/)), and why a headline tokens/sec number [hides more than it reveals](/blog/tokens-per-second-what-it-hides/) unless you know which regime produced it.

The diagnostic sequence that closes this case file works for most of the series: state which resource you believe is saturated, compute the roofline numbers for your operating point by hand (the crossover math above takes 10 minutes), then profile to check that the kernels executing are the ones you think are executing. In this case the arithmetic says INT4 should pay below batch ~80 on this model and GPU. The profile says whether the fused kernel is actually running. When both check out and throughput still doesn't move, you have learned something more interesting: the bottleneck is somewhere you haven't looked yet, most often the KV cache or the scheduler.

## Conclusion

- Weight-only INT4 (GPTQ/AWQ) reduces weight bytes only. The math stays FP16. It pays off in weight-bound regimes and, by construction, cannot accelerate compute-bound ones. Know your bound before you quantize.
- The fallback failure mode is worse than neutral: dequantize-then-GEMM writes and re-reads full FP16 weights, moving ~2.3x the baseline's HBM traffic. Verify fused kernels (Marlin, AWQ, TensorRT-LLM engines) by reading kernel names in Nsight, not by trusting the config.
- Do the crossover math for your own deployment: weight-read floor = bytes/bandwidth, compute time ≈ 2·params·batch/FLOPS. For a 7B on H100 the ideal crossovers land near batch 82 (INT4) and 296 (FP16). Real kernels put them lower. Above the crossover, quantize for capacity, not speed.

### Sources

- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," arXiv 2022. https://arxiv.org/abs/2210.17323
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," arXiv 2023. https://arxiv.org/abs/2306.00978
- Frantar et al., "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models," arXiv 2024. https://arxiv.org/abs/2408.11743
- Marlin kernel repository, IST-DASLab. https://github.com/IST-DASLab/marlin
- NVIDIA TensorRT-LLM repository (weight-only quantization support matrix). https://github.com/NVIDIA/TensorRT-LLM
- NVIDIA H100 Tensor Core GPU specifications. https://www.nvidia.com/en-us/data-center/h100/

*Part of the [LLM Inference & Serving](/series/llm-serving/) learning path. Browse its published articles by topic.*
