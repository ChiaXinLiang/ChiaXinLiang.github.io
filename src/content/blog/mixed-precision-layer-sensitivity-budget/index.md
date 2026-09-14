---
title: "Mixed Precision: Layer Sensitivity and a Deployment Budget"
description: "Mixed precision treats numerical representation as a resource allocation problem."
pubDate: "2026-09-13"
updatedDate: "2026-09-13"
series: "efficient-ml"
code: "eml-7"
order: 7
topic: "Quantization"
level: "intermediate"
tags: ["optimization", "ai-infrastructure"]
heroImage: './section-overview.png'
---

## Overview

![Concept overview: Mixed Precision: Layer Sensitivity and a Deployment Budget](./section-overview.png)

Mixed precision treats numerical representation as a resource allocation problem. Instead of giving every layer the same number of bits, it spends precision where approximation is expensive and removes it where the model tolerates the change. The useful outcome is a deployable checkpoint that meets a quality requirement under a real resource budget.

This article builds that allocation problem up from layer sensitivity to memory accounting and hardware support. A low average bit width is only an intermediate result. The selected formats must still run efficiently, preserve important behavior, and fit together in a supported graph.


*An original conceptual illustration. Numerical plots and examples are illustrative unless explicitly identified as measured evidence.*

## Deep dive

### 1. Define the allocation variables

![Deep-dive illustration: Define the allocation variables](./deep-dive.png)

Consider L quantizable layers. Layer l holds n_l stored weights and gets a precision choice b_l from a supported set. The set might contain 4, 8, and 16 bits, but that availability must come from the backend, not a theoretical wish list.

$$
\mathbf b=(b_1,\ldots,b_L),\qquad b_l\in\mathcal B_l.
$$

The sets can differ by layer. An embedding, a projection, and a normalization operation need not support the same formats. Keep unsupported choices out of the search space instead of hoping an exporter will repair them later.

Storage precision also differs from accumulation precision. A layer can read packed weights while accumulating in a wider datatype. Record both in the configuration, so the allocation is a concrete numerical contract.

### 2. Start with complete memory accounting

![Deep dive: 2. Start with complete memory accounting](./deep-dive-component-03.png)

Weight payload is roughly the sum of element counts times their selected bit widths. Real artifacts also carry scales, zero points, alignment padding, metadata, and any tensors kept in a wider format.

$$
M_{\mathrm{weights}}(\mathbf b)=\sum_{l=1}^{L}\frac{n_l b_l}{8}+M_{\mathrm{scales}}(\mathbf b)+M_{\mathrm{padding}}(\mathbf b).
$$

Runtime memory adds activations, cache state, temporary workspace, and framework allocations. These terms can depend on the chosen kernels as well as the workload. A payload calculation alone cannot prove that a serving process fits.

Say whether the budget constrains serialized bytes, resident model allocation, or peak runtime allocation. All are useful, but swapping one for another produces misleading capacity claims. The runtime measurement that matters uses the intended context length and concurrency.

### 3. Define the quality constraint

Let Q be a task-quality measure and Q_min the acceptance threshold. An allocation can then minimize a resource objective while keeping acceptable behavior.

$$
\min_{\mathbf b\in\mathcal B} M(\mathbf b)\quad\text{subject to}\quad Q(\mathbf b)\ge Q_{\min}.
$$

This expression hides an expensive evaluation problem. Running a full task suite for every combination is rarely practical. Sensitivity estimates guide the search; complete candidate evaluations decide whether the actual constraint holds.

Quality can also mean several conditions at once. Aggregate accuracy can pass while long-context reasoning or a sensitive subgroup degrades. Write those requirements down. A single scalar proxy is convenient for optimization but should not erase application-specific failures.

### 4. Derive a local sensitivity surrogate

Let delta theta be the parameter perturbation quantization introduces. A second-order expansion of the loss gives a local model of the resulting change.

$$
\Delta\mathcal L\approx g^\top\delta\theta+\tfrac12\delta\theta^\top H\delta\theta.
$$

Here g is the loss gradient and H its Hessian at the reference parameters. Near a well-optimized point the gradient term may be small, but check that assumption; it is not a universal identity.

The quadratic term explains why identical weight errors can have different consequences. Perturbations along high-curvature directions cost more under the local loss geometry. Hessian-aware quantization research uses this idea to allocate precision, though the practical estimators and approximations vary between methods.

### 5. Understand trace-based estimates

![Deep dive: 5. Understand trace-based estimates](./deep-dive-component-01.png)

Model the perturbations as roughly zero-mean with covariance Sigma, and the expected quadratic contribution is one half of the trace of H Sigma. An isotropic approximation within a layer reduces this to its curvature trace times an error variance.

$$
\mathbb E[\Delta\mathcal L]\approx\tfrac12\operatorname{tr}(H\Sigma),\qquad \Sigma_l\approx\sigma_l^2 I.
$$

This motivates sensitivity scores that combine numerical error with curvature. Curvature alone is not enough: a highly sensitive layer with very small quantization error may be safer than another layer with larger error.

The assumptions are approximate. Real quantization error depends on weights, clipping, groups, and activation statistics. A trace estimate is a ranking instrument, not a certificate that the final model will meet the quality threshold.

### 6. Measure layer perturbations directly

Another approach quantizes one layer at a time and keeps the others at reference precision. Measure held-out loss or output reconstruction to estimate the effect of each candidate format.

This is easier to interpret than an opaque importance score, but it still misses interactions. Several individually safe changes can together shift the distributions passed to later layers. Calibration activations collected before quantization can then differ from what the final graph actually produces.

Record the evaluation population and whether inputs come from the original or partially quantized model. Repeat selected tests under deployment-relevant conditions. Do not silently generalize sensitivity measured on short calibration sequences to all lengths and tasks.

### 7. Formulate the discrete budget problem

Suppose e_l(b) estimates the quality cost of giving layer l precision b, and m_l(b) estimates its resource cost. A simplified allocation minimizes the sum of estimated errors under a memory budget.

$$
\min_{\mathbf b}\sum_l e_l(b_l)\quad\text{subject to}\quad\sum_l m_l(b_l)\le B.
$$

This is close to a multiple-choice knapsack problem: each layer contributes exactly one option. Dynamic programming, integer optimization, or a constrained heuristic can explore the tradeoff when the costs are modeled well enough.

The separable objective is a stand-in. Cross-layer numerical interactions and graph execution costs can break additivity. Keep the model honest by evaluating shortlisted allocations in the complete system, not by presenting the optimizer's score as a measured result.

### 8. Work through a small allocation

Take an illustrative model with 3 weight groups containing 8 million, 2 million, and 1 million parameters. At 8 bits each, their payload totals 11 million bytes. At 4 bits each, it totals 5.5 million bytes, before all representation overhead.

Suppose the first group tolerates 4 bits, while the second needs 8 and the third needs 16 under held-out validation. The mixed payload is 4 plus 2 plus 2 million bytes, or 8 million bytes. Its parameter-weighted average precision is approximately 5.82 bits.

That average describes payload efficiency, not an executable datatype. The checkpoint still uses 3 concrete formats. It needs compatible kernels, supported boundaries, and enough remaining memory for scales and runtime state. No quality scores or device timings are assumed in this example.

### 9. Compare marginal upgrades

A precision upgrade has a resource cost and an estimated error reduction. Divide the reduction by the added bytes and you get a marginal benefit score that can guide a greedy search.

$$
r_l=\frac{e_l(b_{\mathrm{low}})-e_l(b_{\mathrm{high}})}{m_l(b_{\mathrm{high}})-m_l(b_{\mathrm{low}})}.
$$

The score works when upgrades are independent and the objective tracks the deployment requirement. It does not solve the discrete problem exactly. An expensive upgrade can interact with another choice, and the remaining budget may not fit the highest-ranked option.

Keep several candidate allocations near the budget boundary. Compare them after export and full evaluation. A small surrogate advantage can vanish once packing overhead, unsupported shapes, or quality interactions enter.

### 10. Replace memory with measured latency carefully

Latency-aware allocation needs timings for supported layer configurations on the target device. The cost depends on dimensions, batch size, datatype, kernel selection, and surrounding operations.

Adding up isolated layer timings can underestimate graph-level effects. Format transitions can add conversion, materialization, and dispatch overhead. Kernel fusion can make a pair of operations cheaper together than apart. Two allocations with identical memory can therefore have different latency.

Use measurements or a validated predictor for the search objective, then verify complete graphs. Say whether the target is prefill throughput, decode latency, or another workload. An allocation tuned for large matrix multiplication may be a poor fit for small-batch decoding.

### 11. Treat boundaries as part of the design

Switching formats between neighboring operations can create extra work. Sometimes the conversion folds into a supported kernel; sometimes it needs an intermediate tensor and another launch.

Precision islands, where compatible operations share a representation, can simplify execution. This changes the allocation unit from a single layer to a block or graph segment. The search space shrinks and may match backend behavior better.

The same principle applies to residual additions and shared tensors. Branches that reconverge need compatible numerical interfaces. Verify the exported graph; do not infer compatibility from a layer list. A mixed-precision plan is incomplete until the actual conversions and accumulators are known.

### 12. Separate selection from final evaluation

Use calibration data to estimate ranges and sensitivities, selection data to compare allocations, and held-out data to judge the chosen artifact. Select on the final test population often enough and the allocation overfits to it.

Keep the reference checkpoint, tokenizer, prompts, generation policy, and quality implementation fixed. Differences in these settings can exceed the numerical effect under study. Run diagnostic slices as well as aggregate metrics when the application needs them.

Report the allocation explicitly, including the components kept at high precision. Average bits, compressed file size, and one quality number are not enough to reconstruct the configuration. Layer or block assignments are the concrete evidence replication needs.

### 13. Inspect feasibility before searching

A search can burn a lot of preparation time on configurations that cannot execute. Check the backend's supported weight formats, grouping rules, shapes, and graph boundaries first. Build and run a small representative case for each admissible option.

Drop choices that silently fall back to the reference implementation when the objective requires an accelerated path. Record the fallback policy, because correctness and efficiency have different acceptance conditions.

Also inspect the irreducible memory terms. If cache and workspace already consume most of the capacity, tuning weight precision alone may not solve the problem. Use the full memory model to decide whether another optimization category is needed before launching an expensive allocation search.

### 14. Interpret the frontier rather than one winner

![Deep dive: 14. Interpret the frontier rather than one winner](./deep-dive-component-04.png)

Each feasible artifact has a quality, memory, and latency operating point. A Pareto frontier holds the configurations where improving one measured objective costs another, within the tested set.

Which point to pick depends on application constraints. A quality-sensitive service can spend more memory. A capacity-limited deployment may accept a larger numerical change. The frontier should carry its dataset, hardware, workload, and uncertainty, not pose as a universal ranking.

Mixed precision is most informative when it connects theoretical sensitivity to concrete allocation and execution evidence. The equations explain why precision should be spread unevenly. Only the exported artifact, held-out evaluation, and target-system measurement show whether that distribution is useful.

### 15. Examine uncertainty in sensitivity rankings

![Deep dive: 15. Examine uncertainty in sensitivity rankings](./deep-dive-component-02.png)

Sensitivity estimates are themselves noisy. Changing the calibration sample can shift observed activation ranges and loss gradients, especially for rarely activated features. A trace estimate based on stochastic probes adds estimator variance. Two layers with nearly identical scores are not reliably ordered without more evidence.

Repeat a small number of important estimates on independent calibration subsets. Check whether the ranking stays stable and whether the selected allocations stay feasible. The goal is to find fragile decisions near the resource boundary, not to resample every parameter exhaustively.

A robust allocation can keep extra precision for uncertain components when the budget allows. Or carry several candidates to full evaluation instead of trusting one precise-looking score. This turns statistical uncertainty into a practical engineering choice.

## Conclusion

Finally, keep uncertainty in the quality surrogate separate from variability in runtime measurement. More timing repetitions will not repair a biased calibration population, and more calibration examples will not remove device scheduling noise. Report each limitation with the evidence it affects. No GPU benchmark was executed for this article; all numerical accounting examples are illustrative.

### Sources

- [HAWQ: Hessian Aware Quantization of Neural Networks](https://arxiv.org/abs/1905.03696).
- [HAWQ-V2: Hessian Aware Trace-Weighted Quantization](https://arxiv.org/abs/1911.03852).
- [HAWQ-V3: Dyadic Neural Network Quantization](https://arxiv.org/abs/2011.10680).
- [HAQ: Hardware-Aware Automated Quantization](https://arxiv.org/abs/1811.08886).
