---
title: "QLoRA: Quantized Base Weights and Adapter Numerics"
description: "QLoRA combines a frozen quantized base with trainable low-rank adapters."
pubDate: "2026-09-13"
updatedDate: "2026-09-13"
series: "efficient-ml"
code: "eml-11"
order: 11
topic: "Distillation and Adaptation"
level: "intermediate"
tags: ["optimization", "ai-infrastructure"]
heroImage: './section-overview.png'
---

## Overview

![Concept overview: QLoRA: Quantized Base Weights and Adapter Numerics](./section-overview.png)

QLoRA combines a frozen 4-bit base with trainable low-rank adapters, which cuts the storage the pretrained weights need while keeping a wider computation path for adaptation, and the method also tackles quantization-constant overhead and transient optimizer-memory pressure, so each of those 3 mechanisms affects a different term in the training resource model.

The phrase 4-bit finetuning can hide that separation. The base weights are stored in a low-bit representation, but the original QLoRA setup does not update every packed base value as an ordinary 4-bit trainable parameter. This article derives the numerical interface and explains what you must count, validate, and export.


*An original conceptual illustration. Numerical plots and examples are illustrative unless explicitly identified as measured evidence.*

## Deep dive

### 1. Separate base storage and adaptation

![Deep-dive illustration: Separate base storage and adaptation](./deep-dive.png)

Let Q(W_0) denote the packed quantized base and D its dequantization operator. Low-rank factors A and B add a trainable update to the reconstructed base mapping.

$$
y=\left[D(Q(W_0))+sBA\right]x.
$$

The base representation stays fixed during the adapter training considered here. Only the 2 low-rank factors A and B receive optimizer updates. The scalar s follows the chosen adapter convention and must accompany the artifact.

The original paper uses a low-bit storage datatype and usually BFloat16 computation, other implementations can use different supported paths, and so you should report the actual storage and compute formats rather than describing a whole training graph with one nominal bit width.

### 2. Explain nonuniform code values

![Deep dive: 2. Explain nonuniform code values](./deep-dive-component-04.png)

Uniform affine quantization spaces reconstructed values evenly over a range, while NormalFloat uses a nonuniform set of representable values motivated by a normal-distribution model for weights, placing more resolution where that modeled distribution carries more probability.

Let Phi inverse denote the standard normal quantile function. Quantile-derived values motivate the codebook construction, with normalization and special handling to represent zero in the actual NF4 datatype.

$$
v_i\propto\Phi^{-1}(u_i),\qquad 0<u_i<1.
$$

This schematic equation explains the distributional idea rather than reproducing the exact codebook algorithm, which the primary paper specifies along with its values: a 4-bit code selects one of 16 representable entries, and it is not a conventional floating-point value with arbitrary exponent and mantissa fields.

### 3. Keep the distribution assumption visible

![Deep dive: 3. Keep the distribution assumption visible](./deep-dive-component-03.png)

A normal-inspired codebook such as NF4 is useful when the normalized weight population resembles the assumed distribution, but actual tensors can have outliers, skew, or different local structure, and block normalization also changes the distribution presented to the codebook.

The paper's optimality language for NF4 belongs to its stated distributional criterion. Do not expand it into a universal guarantee of minimum task loss or minimum reconstruction error for every possible tensor. Quantizer design and downstream model quality are different objectives.

Inspect actual block ranges and held-out behavior, and compare the supported 4-bit codebooks under a controlled representation and training recipe when the choice matters, because a numerical datatype's theoretical motivation is evidence for a design while deployment suitability still needs checkpoint-specific evaluation.

### 4. Define blockwise reconstruction

A block of weights shares a scale c under a chosen grouping policy. Each packed code k_i selects one of the 16 codebook values v_k. Reconstruction multiplies that value by the block scale.

$$
\widehat w_i=c\,v_{k_i},\qquad k_i\in\{0,\ldots,15\}.
$$

The scale, group size, axis, and padding policy form part of the format. Two artifacts both called NF4 can still be incompatible if their packing or scale conventions differ.

Include zero-range and partial-block handling in a tiny correctness test, then compare the packed NF4 reconstruction with a direct reference built from known values, which tests the numerical interface before you look at student training or task quality.

### 5. Count quantization constants

Scales are not free. If each block of g weights stores one 32-bit scale, the scale overhead is 32 divided by g bits per weight, before other metadata.

$$
b_{\mathrm{effective}}=4+\frac{32}{g}.
$$

At an illustrative block size of 64, that totals 4.5 bits per weight. A simple 4-bit payload estimate would miss an extra 0.5 bits per value in this format.

The exact artifact can include other constants, unquantized tensors, and padding. Report serialized and resident bytes separately. A nominal 4-bit payload alone tells you neither, and peak training allocation includes additional state beyond the packed checkpoint.

### 6. Derive double-quantization overhead

![Deep dive: 6. Derive double-quantization overhead](./deep-dive-component-01.png)

Double quantization compresses the first-level quantization constants. In the paper's discussed setup, the first-level scale uses an 8-bit representation, with another 32-bit constant shared across 256 such values.

$$
b_{\mathrm{scale}}\approx\frac8{64}+\frac{32}{64\cdot256}\approx0.127\ \text{bits per weight}.
$$

This reduces the described scale overhead from 0.5 bits per weight by about 0.373 bits. These are representation-accounting figures from the specified group sizes, not a measured universal checkpoint compression ratio.

The second quantization, which stores the first-level scales in 8 bits, introduces another approximation and requires its own constants and reconstruction policy, and the paper also centers scale values before that step, so preserve those details when checking an implementation rather than treating double quantization as simply deleting scale storage.

### 7. Work through a storage estimate

For an illustrative tensor containing one billion quantized values, a pure 4-bit payload occupies 500 million bytes. First-level 32-bit scales at group size 64 add 62.5 million bytes.

Using the discussed double-quantization scale accounting instead adds about 15.87 million bytes, before padding and other representation fields. The resulting total is about 515.87 million bytes for this restricted example.

These decimal byte counts do not describe a full language model or its runtime allocation, because embeddings, other retained tensors, adapters, activations, workspace, and optimizer state can all move a 515.87-million-byte estimate substantially, so state the units and the included categories rather than letting a storage example be mistaken for a capacity benchmark.

### 8. Propagate gradients through the fixed base

Although the 4-bit base is frozen, its mapping can still be needed to propagate gradients to earlier trainable components. For a linear layer with output derivative g_y, the input derivative depends on the effective matrix transpose.

$$
g_x=\left[D(Q(W_0))+sBA\right]^\top g_y.
$$

The packed NF4 base gets no optimizer update just because it takes part in this derivative. Freeze status governs parameter updates; the numerical mapping still contributes to forward and backward computation.

This explains why frozen does not mean computationally absent, since dequantization to BFloat16 and matrix execution remain part of every training step, so confirm that base gradients and optimizer slots are not kept unnecessarily while preserving the required input-gradient path for adapted earlier layers.

### 9. Keep activation memory in the model

Adapter state can be small while long-sequence activations dominate training memory. Quantizing base storage to 4 bits does not remove the need to store or recompute the inputs that backward operations use.

Checkpointing changes that tradeoff by recomputing selected activations. Batch size, sequence length, attention execution, and checkpoint granularity all influence peak allocation. The correct capacity question therefore includes more than the 4.5 bits per weight and a trainable parameter count.

Measure the intended workload and inspect allocation by category where possible, because a reported model size and adapter rank are not enough to reproduce a peak-memory result: include the NF4 group size, the compute precision, the optimizer policy, and the workspace behavior in the training configuration you publish.

### 10. Explain paged optimizer state

The original QLoRA work uses paged optimizers to manage transient memory pressure through a supported Unified Memory mechanism. Optimizer pages can move between host and device memory as required.

Paging does not remove the state or make transfers free. It changes where the state lives and how pressure is handled. Transfer behavior depends on the workload, memory availability, device, and software stack.

Distinguish a configuration that normally fits from one that keeps migrating pages during every update, report host-memory use and preparation time when paging affects the resource budget, and do not generalize the paper's stated experiments into a guarantee that paging carries no performance cost under every operating condition.

### 11. Select adapters and baselines consistently

The number and placement of adapters materially affect the trainable update space. The QLoRA paper discusses adapter placement and baseline tuning as important experimental choices under its studied configurations.

A comparison with fewer adapted modules can confound the 4-bit base precision with capacity. Document all selected matrices, ranks, scale conventions, and any additional trained tensors. Match the intended adaptation task and evaluation setup.

Compare with a wider-precision adapter baseline, BFloat16 for instance, and when appropriate a well-tuned full-adaptation baseline. The purpose is to identify what the compressed base changes under fair evidence, rather than attributing every difference to one datatype while the training recipe also changes.

### 12. Evaluate the reconstructed base first

![Deep dive: 12. Evaluate the reconstructed base first](./deep-dive-component-05.png)

Before adapter training, compare the NF4 reference with the original checkpoint. This reveals the initial numerical change that adaptation must work with. Track task quality as well as diagnostic reconstruction error where useful.

Adapter learning can compensate for some effects without restoring every behavior, and a gain on one adaptation task does not prove capability is unchanged elsewhere, so evaluate relevant held-out slices, including the contexts and tasks that matter to the deployment.

Keep preprocessing, prompts, decoding, and quality implementation fixed. A comparison with different generation budgets can obscure the effect of base quantization. No adaptation run or GPU measurement was performed for this article; the examples are numerical accounting and mechanism explanations.

### 13. Choose an export representation

One deployment option keeps the 4-bit base and separate adapters. Another reconstructs the base, adds the update, and writes a new representation. If that representation is quantized again, it introduces a new numerical approximation.

$$
W_{\mathrm{export}}=Q_{\mathrm{new}}\!\left(D(Q(W_0))+sBA\right).
$$

The export operator is not generally equivalent to leaving the original packed base plus a separate adapter branch. Requantization changes assignments among the 16 codebook entries and it changes the block scales, so evaluate the exported checkpoint independently.

Store the original base revision and all adapter metadata, then verify that the serving backend supports the chosen combination, because a training library that loads the artifact is no evidence that a different inference backend supports it or runs it efficiently.

### 14. Relate the method to deployment objectives

QLoRA chiefly addresses adaptation resource requirements through 4-bit frozen storage, small trainable updates, and memory-pressure management. Inference savings depend on what is eventually exported and which kernels execute it.

A student trained with QLoRA can be served in a wider representation such as BFloat16, or its low-bit base can remain part of inference. Those are different deployments. Report the actual path rather than inferring latency from the training method's name.

The complete explanation therefore separates 6 concerns: codebook design, scale overhead, gradient computation, adapter state, paging, and export. That makes the innovation understandable and gives an engineer a concrete list of resource terms to measure for the intended task.

### 15. Verify the layered reconstruction contract

![Deep dive: 15. Verify the layered reconstruction contract](./deep-dive-component-02.png)

Double quantization creates 2 reconstruction stages: recover the first-level scale values, then use them to reconstruct weight values from their codes. A swapped grouping axis or incorrect centering constant can corrupt the result even when the 4-bit payload decodes correctly.

Use small known blocks to verify both stages separately. Include a partial final block and scales with distinct magnitudes. Then compare a tiny adapted linear mapping with a direct reconstructed reference, preserving orientation and the adapter scale.

Numerical agreement establishes implementation correctness within the chosen tolerance, but it does not establish that the normal-inspired NF4 codebook suits every weight population or that the exported student retains all required behavior: held-out evaluation and complete resource measurement answer those additional questions.

Keep the packed base identity, codebook convention, group sizes, compute precision, adapter settings, and export policy together. This metadata is what turns a compact parameter file into a reproducible adaptation artifact.

### 16. Separate persistent and transient allocations

Persistent packed weights and adapter optimizer state remain allocated across many training steps, while temporary reconstructed tiles, attention workspace, and intermediate activations can have much shorter lifetimes, and their overlap determines the peak: adding every individual maximum overestimates memory, and ignoring the overlap underestimates it.

## Conclusion

Inspect the actual allocation timeline when a job fails only on particular sequence lengths, since a rare long batch can create pressure absent from the average workload. Record the maximum tested case and the paging or checkpoint behavior used to handle it, which makes a capacity claim specific to an operating envelope rather than an unsupported claim that a 4-bit checkpoint always fits.

### Sources

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314).
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- [NVIDIA CUDA Unified Memory documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).
