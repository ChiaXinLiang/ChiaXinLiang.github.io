---
title: "Training Offload: CPU, NVMe, Bandwidth, and the Critical Path"
description: "Count state moved to CPU or NVMe, derive transfer and update bounds, and determine which offload work can actually overlap."
pubDate: "Sep 13 2026"
updatedDate: "Sep 13 2026"
heroImage: './section-overview.png'
series: "distributed-training"
code: "train-7"
order: 7
topic: "Parallelism Strategies"
level: "advanced"
tags: ["distributed-training", "ai-infrastructure"]
---

## Overview

![Concept overview: Training Offload: CPU, NVMe, Bandwidth, and the Critical Path. A memory hierarchy cutaway shows GPU HBM, host RAM, and NVMe storage holding different training-state blocks.](./section-overview.png)

Training offload uses host memory or storage to hold state that would otherwise occupy GPU memory. It can make a large model trainable on a smaller accelerator set. It can also free device capacity for activations and microbatches. The memory saving is real, and for the 7-billion-parameter Adam configuration worked below it moves 84 GB of optimizer state off the accelerators, but the displaced bytes still have to be stored, transferred, and sometimes updated elsewhere.

The engineering problem is to place those operations on a dependency timeline: gradients cannot be transferred before they are produced, updated parameter values cannot return before the optimizer has computed them, and a parameter shard needed for the next layer cannot arrive after that layer has already stalled waiting for it.

We will distinguish optimizer offload from parameter offload, derive transfer and CPU-bandwidth bounds, and work an illustrative mixed-precision Adam example, whose numbers are explicit assumptions rather than measured hardware results, so replace them with the effective bandwidth and state sizes of the actual machine before making a capacity or throughput decision.

## Deep dive

### 1. Identify exactly what is being offloaded

Optimizer offload places selected optimizer state and possibly optimizer computation in host memory, and for Adam the first and second moments and any master-weight copy can be large, so keeping them off the GPU removes an important persistent-state contribution while compute weights stay available for forward and backward.

Parameter offload places some parameter values outside GPU memory between uses. The execution schedule must fetch the values needed by an upcoming operation and release them when safe, and combining parameter sharding with offload can reduce device retention further while adding materialization and transfer dependencies.

Activation offload is a different mechanism: saved tensors move to another memory tier and return when backward needs them. It trades host capacity and transfers against saved GPU activation memory. Activation recomputation instead recreates those values with extra arithmetic. To compare the 2, you need both transfer cost and replay cost.

NVMe offload adds a storage tier beyond host DRAM. It increases available capacity but adds 3 things: another service path, a buffering requirement, and a latency distribution. A storage device’s advertised sequential bandwidth is not automatically what the application achieves. Access sizes, concurrency, filesystem behavior, and a shared workload all change it.

### 2. Derive a state-placement budget

Let P be logical parameter count, D the state-sharding degree, and o bytes of optimizer-related state per parameter. Ideal optimizer-state device retention falls by P o divided by D when that local shard moves fully to host memory, and host retention increases by the corresponding amount.

For an illustrative mixed-precision Adam configuration with 4-byte master weights and 2 separate 4-byte moments, o equals 12. A 7-billion-parameter model has 84 GB of this state before sharding. With D equal to 8 and balanced ownership, the local shard is about 10.5 GB per rank.

Those numbers are not a complete host-memory budget, because gradient buffers, parameter staging, pinned transfer buffers, application workers, page cache, runtime libraries, and checkpoint activity can coexist, and because several ranks on 1 host share physical DRAM capacity, which means you must multiply per-rank shards and add common services.

Likewise, the device still needs compute weights or the materialized working set. It needs gradients according to the schedule, plus activations and workspaces. Offloading 1 category does not remove the others. Count the largest concurrent allocation in both tiers. Smaller persistent GPU state does not mean capacity is solved.

### 3. Count transfers in both directions

![Deep-dive illustration: Count transfers in both directions](./deep-dive.png)

For a transfer of S bytes over a path with effective throughput beta, an optimistic service bound is

$$
T_{\mathrm{copy}}\ge\frac{S}{\beta}.
$$

Startup, synchronization, registration, packing, and contention can make the observed duration larger, so the bound must use the relevant direction and path, and you cannot substitute bidirectional headline bandwidth for the one-direction rate of a single transfer.

Suppose 7 billion gradients use 2 bytes each and 7 billion updated compute weights also use 2 bytes each. Moving the gradients out and the weights back transfers 28 GB in total. At an assumed effective 25 GB/s, a serialized two-direction copy budget is at least 1.12 seconds.

A sharded implementation may transfer only local owned slices or use a different data representation, and some systems overlap outgoing and incoming chunks, so that 28 GB figure is a deliberately simple accounting baseline. Inspect the actual tensors and ordering to find the byte volume and concurrency used by a particular offload method.

### 4. The CPU optimizer has a bandwidth bill

Moving optimizer computation to the CPU does not make it negligible. An Adam update reads gradients, master weights, first moments, and second moments, then writes updated master weights and moments, so each parameter update can create heavy DRAM traffic even before temporary conversions or additional implementation passes.

For the illustrative representation with 2-byte gradients and 4-byte master and moment values, an idealized single-pass read/write count is 26 bytes per parameter: 14 read and 12 written. For 7 billion parameters, that is 182 GB of DRAM traffic.

With an assumed effective optimizer memory bandwidth of 200 GB/s, the traffic alone gives a lower bound near 0.91 seconds. This is not a CPU Adam benchmark. Vectorization, arithmetic, cache behavior, thread placement, casts, and extra passes can all change observed time.

$$
T_{\mathrm{update}}\gtrsim\max\left(\frac{\mathrm{DRAM\ bytes}}{\beta_{\mathrm{DRAM}}},\frac{\mathrm{optimizer\ operations}}{P_{\mathrm{CPU}}}\right).
$$

The expression separates bandwidth and arithmetic limits. A nominally powerful CPU can still underperform if the optimizer reads remote NUMA memory or competes with input workers and several other ranks for the same channels.

### 5. Work the sequential critical path

![Deep dive: 5. Work the sequential critical path](./deep-dive-component-02.png)

In a simple optimizer-offload schedule, backward produces gradients, gradients transfer to the host, the CPU updates its state, and updated compute weights transfer back before the next forward pass. Using the illustrative bounds above, the extra serialized service is at least about 2.03 seconds.

Do not add that 2.03 seconds blindly to every real training step. An implementation can stream gradient chunks, begin CPU updates on ready chunks, and overlap different stages. Conversely, conversion and coordination overhead can make the observed exposed interval larger than the component lower bounds.

For chunk j, define readiness r_j, outgoing transfer duration a_j, CPU update duration u_j, and incoming transfer duration b_j. With 1 serial worker per stage, completion depends on both the preceding stage and the previous chunk occupying that worker. The schedule is a staged pipeline, not a sum of independent operations free to start whenever convenient.

A useful steady-state bound for balanced chunks is the largest stage service time, plus pipeline fill and drain, and the last updated chunk can remain exposed after backward ends. Measure chunk readiness and completion timestamps. They identify which stage limits the pipeline and how much of the tail reaches the next forward pass.

### 6. Parameter offload moves the dependency into layers

![Deep dive: 6. Parameter offload moves the dependency into layers](./deep-dive-component-01.png)

When parameters are not retained on the GPU, each layer or sharding unit needs a fetch before compute. Prefetch can overlap the next unit’s transfer with the current unit’s arithmetic. Too little lookahead exposes transfer latency. Too much materializes several units and uses up the capacity offload was meant to save.

A local memory constraint can be written schematically as the sum of 3 terms: retained device state, active and prefetched units, and the activation and workspace peak. The exact storage can involve aliasing or buffer reuse, so measure lifetimes rather than summing duplicate representations automatically.

Access frequency matters. A parameter unit may be needed 3 times, during forward, during recomputation, and during backward, so keeping it longer can reduce transfers at greater device-memory cost while releasing it aggressively can increase host or storage traffic. Two questions decide the policy. When will the unit be used again, and can another tier supply it in time?

Align offload units with compute and sharding boundaries at first, then refine when traces show dominant stalls or wasted materialization. A policy based only on parameter size can miss expensive repeated accesses created by activation checkpointing or pipeline interleaving.

### 7. NVMe adds capacity and another pipeline stage

Storage-backed offload commonly stages data through host memory before the GPU can use it. The path may include 4 steps: storage reads, host buffering, transfer preparation, and a device copy. Compatible direct-I/O mechanisms can change this path, but check support and behavior for the actual stack.

If a unit of S bytes must move through independent serial tiers, its unoverlapped service includes each tier’s startup and transfer duration, and in steady state pipelining can approach the slowest tier’s throughput as long as enough buffering and concurrency exist, but it cannot exceed that tier just because another link is faster.

Buffer count, request size, alignment, filesystem behavior, and competing checkpoint writes can all change effective storage throughput, and random small requests and large sequential requests represent different workloads, so benchmark the access pattern that the offload scheduler actually produces rather than 1 unrelated disk test.

Storage latency variation also affects tails. A rare slow read can stall the next required layer even when average bandwidth looks high enough. Record percentiles and complete-step variability, and evaluate whether prefetch headroom absorbs those delays under the supported workload.

### 8. NUMA locality and pinned buffers matter

Host memory belongs to a physical topology. A CPU optimizer may run on 1 socket while its state sits on another. That path uses inter-socket bandwidth and sees higher latency. GPU transfers can likewise follow different paths depending on CPU affinity and memory placement.

Place host state, optimizer threads, and transfer buffers deliberately. Pinned memory can support efficient asynchronous copies, but pinned allocations use host resources and are not unlimited, and several ranks sharing 1 root complex can contend even if each individual GPU copy benchmark looks healthy.

Input processing and checkpoint writing use the same host and storage resources. An offload benchmark run without those services can overstate production performance. Include the representative concurrent workload when evaluating the chosen configuration.

Correct synchronization also governs buffer reuse. A host or device staging buffer cannot be overwritten while a transfer or kernel still reads it, and events and completion signals establish those ownership transitions. Avoid fixing lifetime bugs with indiscriminate global synchronization, because that can hide races while destroying the intended overlap schedule.

### 9. Compare feasibility and useful throughput honestly

Offload can be valuable even when fixed-work step time increases: it can make training possible on available resources. Compare feasible configurations with documented device and host capacity. Do not compare against an unoffloaded instance that cannot execute.

Measure 5 quantities: peak memory in every tier, exposed transfer time, CPU update time, complete optimizer-step time, and useful training tokens. Check the optimizer update and numerical representations against a manageable reference. Document precision changes or different loss weighting as changes to the training configuration.

Sweep 4 knobs in small controlled experiments: chunk size, prefetch depth, thread placement, and buffer count. Too-small chunks pay startup and scheduling overhead; too-large chunks delay readiness and increase transient memory. The best setting balances the actual compute, transfer, and update stages.

Finally, test checkpoint and recovery behavior with the offloaded state. A training job that fits and runs steadily can still fail when saving gathers state or when restart recreates temporary buffers differently, so capacity planning should include supported operational paths, not only the middle of a successful iteration.

## Conclusion

Offload changes where training state lives and sometimes where updates execute. It replaces GPU retention with host or storage capacity, transfer service, and new scheduling dependencies. Streaming can hide some work, but the final required state still has a critical path.

Count the bytes in every tier, measure effective throughput under contention, and connect each chunk to its producer and consumer. Evaluate offload as a complete feasible training design rather than a free memory-reduction switch.

### Sources

- [DeepSpeed ZeRO-Offload tutorial](https://www.deepspeed.ai/tutorials/zero-offload/): optimizer offload and CPU update implementation.
- [Ren et al., ZeRO-Offload](https://arxiv.org/abs/2101.06840): heterogeneous training-state placement and computation.
- [Rajbhandari et al., ZeRO-Infinity](https://arxiv.org/abs/2104.07857): memory hierarchy and storage-backed training.
- [PyTorch CUDA notes](https://docs.pytorch.org/docs/stable/notes/cuda.html): asynchronous copies, pinned memory, and stream synchronization.
