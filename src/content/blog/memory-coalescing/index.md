---
title: 'Memory Coalescing: Tiny Code Changes, Massive Speedups'
description: "Why swapping two index variables can change a CUDA kernel's DRAM traffic by 8x, how to count cache lines per warp by hand, and the shared-memory tricks that fix it."
pubDate: 'Sep 12 2026'
heroImage: './cover.png'
code: 'mem-1'
order: 11
series: 'ai-performance'
topic: 'CUDA & Kernels'
tags: [cuda, gpu, memory]
---

One load instruction can move 128 bytes or 1,024 bytes through a GPU's memory system, and both versions compute exactly the same result. The difference is which addresses the 32 threads of a warp ask for, and it is decided by something as small as the order of two loop indices. Get it wrong on a memory-bound kernel and you leave most of your HBM bandwidth on the table; on a matrix transpose, the naive version routinely runs several times slower than the coalesced one on the same silicon. This post is about that mechanism, called memory coalescing, and about how to reason through it with nothing but arithmetic.

If warps, SIMT execution, and why GPUs are throughput machines are new territory, start with [CPU vs GPU](/blog/cpu-vs-gpu-latency-vs-throughput-machines/) and come back; this article assumes that background.

## One instruction, thirty-two addresses

A CUDA streaming multiprocessor does not execute threads one at a time. It executes *warps*: groups of 32 threads that issue the same instruction in lockstep. When a warp hits `float x = A[idx];`, the hardware sees one load instruction carrying 32 addresses, one per thread.

The memory system does not fetch individual bytes either. Global memory traffic moves in 32-byte *sectors*, and cache lines in the L1/L2 hierarchy are 128 bytes, or four sectors. When the warp's load reaches the load/store unit, a coalescer looks at all 32 addresses and asks: how few sectors can service this request?

The best case: 32 threads read 32 consecutive 4-byte floats, aligned to a 128-byte boundary. That is exactly 128 contiguous bytes, four sectors, one cache line. One request, four sectors moved, every byte used. Nsight Compute reports this as the `sectors/request` ratio, and 4.0 is the ideal for 4-byte accesses. This is a *coalesced* access.

The worst common case: each thread reads an address 16 KB away from its neighbor's, which is what happens when adjacent threads walk down a column of a large row-major matrix. Now each of the 32 addresses lands in a different sector on a different cache line. One request, 32 sectors moved, and from each 32-byte sector the warp uses 4 bytes. Nsight shows 32 sectors per request, an 8x traffic amplification, and 12.5% of the moved bytes doing useful work.

Nothing about the arithmetic changed. The instruction count didn't change. Only the *shape* of the addresses did.

![Coalesced versus strided warp access: 32 loads landing in one 128-byte cache line versus 32 separate lines](./fig-coalescing.png)

## A worked example you can do on paper

Take a 4096 x 4096 matrix of fp32, stored row-major. That is 64 MB; one row is 4096 floats, or 16 KB.

**Row-wise read (coalesced).** Thread `t` of a warp reads `A[row * 4096 + col0 + t]`. The 32 addresses are `base, base+4, base+8, ..., base+124`: a contiguous, aligned 128-byte span.

- Cache lines touched per warp load: **1**
- Bytes moved: 128. Bytes used: 128. Efficiency: **100%**
- Reading the full matrix moves 64 MB of DRAM traffic, the theoretical minimum.

**Column-wise read (strided).** Thread `t` reads `A[(row0 + t) * 4096 + col]`. Adjacent threads are now 4096 floats apart, so consecutive addresses differ by 16,384 bytes.

- Cache lines touched per warp load: **32**
- Bytes moved: 32 sectors x 32 B = 1,024. Bytes used: 128. Efficiency: **12.5%**
- Reading the full matrix this way moves 512 MB for the same 64 MB of useful data.

Now put bandwidth numbers on it. An H100 SXM delivers about 3.35 TB/s of HBM3 bandwidth per NVIDIA's datasheet. The coalesced scan has a floor of 64 MB / 3.35 TB/s, roughly 19 microseconds. The strided scan's floor is 512 MB / 3.35 TB/s, about 153 microseconds, and the measured gap is usually worse than 8x because thirty-two independent line fetches per warp also burn issue slots, MSHR entries, and DRAM row activations that the tidy version never needed. The kernel didn't get more math. It got more traffic.

One more paper exercise worth doing: shift the coalesced access by a single float, so the warp reads `base+4` through `base+128`. The span now straddles a line boundary and touches five sectors instead of four. That misalignment penalty is real but mild (25% extra traffic), which is why alignment is a second-order concern next to stride, the thing that can cost you 8x.

## Going deeper: the toolbox that restores coalescing

Knowing the failure mode, the classic fixes all become one idea: *reshape the access so that the warp's 32 addresses are contiguous when they hit DRAM, and absorb any awkward stride somewhere cheaper.*

**Vectorized loads.** If each thread reads a `float4` (16 bytes) instead of a `float`, the compiler emits a single 128-bit `LDG.128` instruction and the warp moves 512 bytes, four full cache lines, per load instruction. The bytes were coalesced either way; what you save is instruction issue and address-generation overhead, which matters in kernels that are instruction-bound rather than purely bandwidth-bound. NVIDIA's developer blog measured meaningful throughput gains from exactly this change in otherwise identical copy kernels. The requirement is 16-byte alignment, which `cudaMalloc` guarantees for the base pointer; your indexing has to preserve it.

**Tiling through shared memory.** Some access patterns are strided no matter how you write them; a transpose must read rows and write columns, or vice versa. The fix is to split the awkward stride across two memory spaces. Each thread block copies a 32 x 32 tile from global memory into shared memory with coalesced row-wise reads, synchronizes, then writes the transposed tile back with coalesced row-wise writes to the destination. The column-wise access still happens, but *inside shared memory*, which is on-chip SRAM with no notion of DRAM sectors. Global memory only ever sees contiguous 128-byte transactions in both directions.

**Padding away bank conflicts.** Shared memory has its own granularity: 32 banks, each 4 bytes wide, and a warp achieves full speed only when its 32 accesses land in 32 different banks. A `tile[32][32]` array puts every element of a column in the same bank (32 mod 32 = 0), so reading a tile column serializes into a 32-way bank conflict, thirty-two round trips where one should do. Declare the tile as `tile[32][33]` and each row starts one bank later than the previous row, so a column now spans all 32 banks. One wasted column of padding, 128 bytes per tile, buys back a 32x serialization. This exact pair of fixes is the canonical transpose optimization in NVIDIA's shared-memory material.

![Shared-memory tiling for transpose: coalesced global reads and writes, with a 33-wide tile to eliminate bank conflicts](./fig-tile-padding.png)

**TMA: hardware takes over the copy.** On Hopper and Blackwell, the Tensor Memory Accelerator is a dedicated copy engine per SM that moves multidimensional tiles between global and shared memory from a single descriptor: base address, tensor shape, tile size. One thread issues the copy; the TMA hardware computes all the addresses, handles out-of-bounds edges, and streams the tile asynchronously while the warps compute on the previous one. The register and instruction cost of address arithmetic, a real tax in older pipelined-copy code, drops to nearly nothing. This is coalescing as a hardware service: the descriptor tells the engine the layout, and the engine generates optimal transactions. CUTLASS and Triton lean on it heavily for Hopper-class GEMM and attention kernels.

## Common misconceptions

**"Threads must access memory in thread order for coalescing to work."** Not since the Fermi generation. The coalescer cares about which sectors the 32 addresses cover, not which thread asked for which. Thread 0 reading the last float of a 128-byte span and thread 31 reading the first coalesces into the same single line fetch as the tidy ordering. Permutations are free; *spread* is what costs.

**"The cache will absorb an uncoalesced pattern."** Only if the data gets reused before eviction, and for a streaming pass over a 64 MB matrix it does not. L2 on an H100 is 50 MB; a strided full-matrix read touches each of the 512 K cache lines once, uses 4 bytes per sector visit, and would need the line to survive 4096 intervening line fetches to serve the next hit. It won't. Caches rescue *locality*, and a giant stride is the definition of not having any. The sector traffic to DRAM happens regardless.

**"Coalescing is a hand-written-CUDA concern; frameworks make it irrelevant."** Data layout decides coalescing, and layout is set at the framework level. It is why PyTorch has `.contiguous()` and channels-last memory formats, why structure-of-arrays beats array-of-structures for GPU-resident data, and why a transposed weight matrix can silently change a GEMM's performance class. The kernel author may be cuBLAS, but the tensor strides that kernel receives are yours.

## Where this sits in the bigger picture

Coalescing is the [memory wall](/blog/the-memory-wall-latency-numbers/) at its finest granularity. At the datacenter scale you buy bandwidth with [HBM stacks](/blog/from-dram-to-hbm/); at the kernel scale you can waste 87.5% of it with one bad subscript. Modern inference workloads are dominated by memory-bound phases, so the difference between 4 and 32 sectors per request is not an academic detail; it is the same lever that made [one kernel change move DeepSeek's API prices](/blog/when-a-kernel-cuts-api-prices/). It is also the first thing LLM-generated kernels get wrong: much of the gap that [a year of KernelBench](/blog/a-year-of-kernelbench/) measured between generated and expert kernels comes down to models that know CUDA syntax but not sector arithmetic. When you profile, this is the metric to open first. Nsight Compute's memory workload section shows sectors per request directly, and a number far above the ideal is the cheapest large speedup you will ever find: no algorithm change, no new hardware, just addresses.

The same instinct extends upward. Tiling into shared memory is the GPU cousin of [cache blocking](/blog/caches-how-locality-rescues-speed/), and TMA is the industry admitting the pattern is so universal it deserves dedicated silicon.

## Verify the improvement at the right memory level

Measure transaction efficiency and elapsed kernel time after changing the layout. Fewer sectors per request can explain an improvement, but cache hits, instruction overhead, and occupancy can also change. A reduction in requested sectors does not imply an identical reduction in bytes reaching DRAM, because the cache may satisfy some requests. Compare cache and DRAM counters rather than treating all memory traffic as one quantity.

Keep array shape, dtype, and work identical between versions. For a transpose, verify every output element, including edge tiles whose dimensions are not multiples of the block size. Then measure representative problem sizes. A fix that helps a large streaming matrix may have little effect on a small matrix resident in cache. The result should establish where the layout change helps and explain that behavior with counters.

## Takeaway

- A warp's load is judged by how many 32-byte sectors its 32 addresses cover: 4 sectors per request is ideal for fp32, and a large stride pushes it to 32, an 8x traffic amplification you can compute by hand before ever profiling.
- The fixes form a ladder: contiguous indexing first, `float4` vectorization for instruction efficiency, shared-memory tiling (with width-33 padding against bank conflicts) for inherently strided patterns, and TMA descriptors on Hopper/Blackwell to hand the whole problem to hardware.
- Layout is destiny even if you never write a kernel: tensor strides chosen in Python decide whether the library kernel underneath runs at 100% or 12.5% memory efficiency.

## Sources

- NVIDIA, *CUDA C++ Best Practices Guide*, memory optimization chapters — https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- NVIDIA, *CUDA C++ Programming Guide*, device memory accesses — https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Mark Harris, "CUDA Pro Tip: Increase Performance with Vectorized Memory Access," NVIDIA Developer Blog — https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
- Mark Harris, "Using Shared Memory in CUDA C/C++," NVIDIA Developer Blog — https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
- NVIDIA, "NVIDIA Hopper Architecture In-Depth" (TMA introduction) — https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- NVIDIA, *Nsight Compute Documentation* (memory workload analysis, sectors/request) — https://docs.nvidia.com/nsight-compute/

*Part of the **AI Performance Engineering** series. Previously: [Goodput: Your "100% Utilized" Cluster Is Mostly Wasted](/blog/goodput-vs-utilization/). Next: occupancy, and why more threads is not always faster.*
