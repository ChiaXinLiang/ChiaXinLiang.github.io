---
title: "Hardware-Aware Model Design: Shapes, Latency, and Search Spaces"
description: "An efficient architecture is efficient on an execution system."
pubDate: "2026-09-13"
updatedDate: "2026-09-13"
series: "efficient-ml"
code: "eml-12"
order: 12
topic: "Architecture Design"
level: "intermediate"
tags: ["optimization", "ai-infrastructure"]
heroImage: './section-overview.png'
---

## Overview

![Concept overview: Hardware-Aware Model Design: Shapes, Latency, and Search Spaces](./section-overview.png)

An efficient architecture is efficient on an execution system. Reducing mathematical operations can help, but the mapping from operations to latency depends on 4 further things: matrix shapes, memory traffic, kernel support, and workload. Hardware-aware model design brings those constraints into the architecture decision instead of checking them only after training.

This article builds a performance model for that decision and explains why equal-FLOP models can behave differently, and the goal is not to predict every timing from a formula but to identify useful design variables, reject misleading proxies, and connect an architecture's quality to measured execution.


*An original conceptual illustration. Numerical plots and examples are illustrative unless explicitly identified as measured evidence.*

## Deep dive

### 1. Define the deployment envelope

Specify the target device, backend, numerical format, batch, input resolution or sequence length, and concurrency. Those 6 settings determine the operations that the architecture actually executes.

For image classification, you might evaluate an architecture on one fixed resolution and batch. For language serving, the 2 phases of prefill and decode produce different matrix shapes and state lifetimes. A single architecture can occupy different performance regimes across those workloads.

Also define the acceptance constraint, choosing among 4 candidates: latency deadline, throughput, peak memory, energy, or some combination of them. Hardware-aware means the resource objective is attached to an operating envelope. Without that envelope, the phrase describes an intention rather than a testable design.

### 2. Count matrix work precisely

![Deep-dive illustration: Count matrix work precisely](./deep-dive.png)

For multiplication of an M-by-K matrix with a K-by-N matrix, dense arithmetic performs about 2MKN floating-point operations under the convention that multiplication and addition each count once.

$$
C_{\mathrm{GEMM}}\approx2MKN.
$$

This count describes mathematical work, not elapsed time. It does not include data movement, launch overhead, padding, conversion, or other operators surrounding the multiplication.

Models with the same 2MKN count can show different tile utilization and parallelism, because a thin dimension limits the available work per tile while a small overall problem leaves much of a large device idle, so keep shapes alongside operation counts in every architecture comparison.

### 3. Estimate arithmetic intensity

Arithmetic intensity divides operations by bytes transferred at a chosen memory boundary. A simple idealized estimate reads each of the 2 input matrices once and writes the output once, using p bytes per element.

$$
I\approx\frac{2MKN}{p(MK+KN+MN)}.
$$

The denominator is a lower-level modeling assumption rather than an exact measurement, because 5 effects can change the traffic it claims to count, cache reuse, repeated reads, accumulation, fusion, and packing, so define which boundary the estimate concerns, device memory rather than registers for instance.

This ratio helps explain why smaller batch or decode shapes can become bandwidth sensitive, and more arithmetic reuse can raise intensity, but only when the execution path realizes that reuse, so use the model to form a hypothesis and then check measured traffic or performance when it is available.

### 4. Apply a roofline bound

![Deep dive: 4. Apply a roofline bound](./deep-dive-component-04.png)

Let P_peak be an appropriate compute ceiling and BW the relevant bandwidth ceiling. A basic Roofline model bounds achievable operation throughput by the smaller of 2 ceilings: compute capability, and bandwidth times intensity.

$$
P\le\min(P_{\mathrm{peak}},BW\cdot I).
$$

The Roofline bound is optimistic. It does not account for every execution bottleneck, and advertised device peaks can require particular formats, instruction paths, or sparsity assumptions. Use the ceiling matching the actual kernel.

An architecture change that reduces bytes can help a bandwidth-sensitive workload even if FLOPs remain similar, while fewer operations may not lower latency when dispatch or memory movement dominates, so the Roofline model identifies the limiting resource rather than declaring operation reduction useless.

### 5. Examine tile padding

![Deep dive: 5. Examine tile padding](./deep-dive-component-03.png)

Suppose a kernel organizes output tiles of t_M by t_N and reduction tiles of t_K. A conceptual padded operation count rounds each of the 3 dimensions up to a multiple of its tile size.

$$
C_{\mathrm{padded}}\approx2\left\lceil\frac M{t_M}\right\rceil t_M\left\lceil\frac N{t_N}\right\rceil t_N\left\lceil\frac K{t_K}\right\rceil t_K.
$$

Actual kernels can use masked edges, specialized variants, or different scheduling, so this is a utilization illustration rather than a universal implementation formula.

The expression shows why small changes in width can have discontinuous effects, since all 3 dimensions round up: removing a few channels might reduce mathematical work while leaving the same tile footprint, and a larger structured change can cross a shape boundary and have a much bigger effect on execution. Verify the target kernel rather than assuming a smooth FLOP-to-latency relationship.

### 6. Work through shape utilization

With an illustrative output tile width of 64, an output dimension of 65 spans 2 tiles along that axis. The conceptual tiled width is 128, giving about 50.8 percent useful columns under this simplified view.

Reducing the dimension to 64 fits one tile and can change the utilization regime. Reducing it only to 64.5 is not a meaningful integer channel count, and reducing another dimension without changing this edge does not automatically remove the extra tile.

This example does not predict a speedup of exactly 2. Kernel selection, occupancy, memory, and launch overhead still matter. It explains why you should test architecture widths against supported execution shapes rather than pick them solely from a continuous parameter-count target.

### 7. Model launch and boundary costs

A simple graph timing model adds operator execution, dispatch, and boundary work. Its terms can interact when fusion changes the graph.

$$
T_{\mathrm{graph}}\approx\sum_i T_i+T_{\mathrm{dispatch}}+T_{\mathrm{conversion}}+T_{\mathrm{materialization}}.
$$

Many tiny operators can pay the 3 overhead terms above, dispatch, conversion, and materialization, out of proportion to their mathematical work. An architecture with fewer large compatible blocks can therefore execute differently from one with many small heterogeneous branches.

Fusion can avoid writing and rereading intermediates, but it depends on backend support and graph structure, so inspect the exported graph and the compiled execution: a symbolic operator list does not tell you which boundaries remain during deployment.

### 8. Include activation and state memory

The 4 properties of architecture width, depth, token count, and resolution all affect intermediate memory. Stateful language inference also stores cache or recurrent state whose size depends on model structure and context policy.

A parameter-efficient architecture can still have expensive activations. High-resolution features or long token sequences can dominate memory traffic and peak allocation. Reducing model weights does not directly solve those terms.

Define tensor lifetimes and the workload when estimating peak memory, since summing every tensor can overestimate if lifetimes do not overlap while counting only the single largest activation can underestimate simultaneous buffers, so measure the complete artifact when capacity is a deployment constraint.

### 9. Treat operator support as a constraint

An architecture can select operations that are theoretically efficient but poorly supported on the target backend. A depthwise convolution, an unusual activation, a sparse pattern, and a dynamic selection rule are 4 cases that may take a very different execution path across devices.

Build representative microbenchmarks for the admissible operations and shapes before expensive model training. Check whether unsupported choices trigger decomposition or fallback. Keep the correctness tests separate from resource acceptance.

The useful search space includes operators that can be exported and executed under the intended numerical policy. Excluding an unsupported option is a practical design decision, not proof that the mathematical operation is inefficient on every possible system.

### 10. Use latency lookup tables cautiously

A lookup table can store measured operator timings indexed by 3 keys: shape, format, and device. A search algorithm can then estimate candidate cost without executing every complete model during selection.

This approach is useful when the table covers the search space and operator costs approximately compose. The estimate weakens under any of 3 conditions: interpolation outside measured shapes, graph fusion, and memory interactions.

Validate shortlisted complete architectures and check predictor error near the constraint boundary, because a candidate predicted to sit just below a deadline may be infeasible once exported, and keep measurements, backend versions, warmup, and repetition policy with the table so that it is reproducible evidence rather than an unexplained constant database.

### 11. Connect width and depth to quality

Reducing width or depth changes capacity and computation. Resolution and token reduction change the information presented to the model. Explore those 4 variables with quality-resource tradeoffs rather than only one cost metric.

A useful design objective runs in 1 of 2 directions: minimize latency under a quality threshold, or maximize quality under a resource budget. The architecture parameters are discrete and interact with training, so a proxy score needs complete validation.

Compare candidates under documented training budgets, because an architecture trained longer or on more data can look better for reasons unrelated to its hardware mapping, and preserve the reference recipe and report any additional preparation cost needed to obtain the efficient candidate.

### 12. Understand device-specific rankings

Two devices can favor different operator types, widths, and numerical paths. Even on one device, changes in batch or input length can reverse a ranking. Hardware-aware architecture research such as MnasNet, ProxylessNAS, and FBNet connects selection to a resource model or target execution rather than relying only on FLOPs.

Those 3 papers' specific experiments belong to their stated devices and tasks, and their broader lesson is that resource feedback should match the deployment goal, not that one published architecture is universally optimal.

When transferring an architecture to another backend, repeat complete measurements and inspect support. Similar advertised compute peaks do not mean equal operator performance or the same ordering between candidates.

### 13. Build a controlled comparison

Choose a baseline and several candidates that change one interpretable design variable at a time where feasible. Record 6 quantities: parameter count, operation count, activation or state bytes, measured latency, throughput, and task quality.

Use the same preprocessing and numerical contract. Warm up execution and report repetitions with an appropriate variability summary. Separate initialization and compilation from steady-state inference unless startup is itself the objective.

No device microbenchmark or model-training experiment was performed for this article. The shape examples and the Roofline bounds are explanatory. A deployment decision requires executing the real candidate artifacts on the intended system, with quality evaluated under a consistent task protocol.

### 14. Make co-design concrete

![Deep dive: 14. Make co-design concrete](./deep-dive-component-01.png)

Co-design can modify the model to fit an execution primitive or improve the execution system to support a valuable model structure. Both require an explicit interface with 4 parts: shapes, numerical representation, memory layout, and scheduling assumptions.

A regular block structure can be easier to compile than unpredictable data-dependent selection. Dynamic algorithms can still be useful, but selection overhead and workload variability belong in their measured cost.

The most informative explanation connects an architectural choice to a resource mechanism and then to evidence, because equal FLOPs do not imply equal execution and a favorable microbenchmark does not imply equal complete-model quality, so hardware-aware design becomes useful when those 3 levels are connected rather than collapsed into one efficiency score.

### 15. Interpret measurement uncertainty at the boundary

![Deep dive: 15. Interpret measurement uncertainty at the boundary](./deep-dive-component-02.png)

Timing variability can come from 5 sources: device clocks, background work, allocation, scheduling, and input-dependent execution. A search that accepts a candidate from one unusually fast run can violate a real service deadline.

Use a measurement policy that matches the requirement, since a throughput target needs sustained workload evidence while a latency service objective can require tail measurements under realistic concurrency, and isolated operator timing answers a different question from end-to-end request latency.

Carry uncertainty into selection near the budget boundary, where a small margin can be preferable to a candidate whose predicted advantage is smaller than the model or timing error, and treat that as a practical consequence of making the resource constraint an operating requirement.

Finally, preserve the measured backend and artifact revision. Compiler changes can improve one graph while leaving another unchanged. Reproducible hardware-aware design therefore records the software mapping as well as the model architecture and device identity.

## Conclusion

A final comparison should keep the candidate's quality-resource point attached to its evidence, so store 5 things together: the architecture configuration, the exported graph, the numerical policy, the workload, and the measurement summary. That package allows another engineer to test whether the same tradeoff holds after a device or compiler change, and it keeps a favorable operation count from being reused as if it were an independently measured latency result.

### Sources

- [Roofline: An Insightful Visual Performance Model](https://doi.org/10.1145/1498765.1498785).
- [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626).
- [ProxylessNAS](https://arxiv.org/abs/1812.00332).
- [FBNet: Hardware-Aware Efficient ConvNet Design](https://arxiv.org/abs/1812.03443).
