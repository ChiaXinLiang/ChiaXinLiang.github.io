---
title: "LoRA: Low-Rank Updates and Training-State Memory"
description: "Low-rank adaptation changes which parameters are trained."
pubDate: "2026-09-13"
updatedDate: "2026-09-13"
series: "efficient-ml"
code: "eml-10"
order: 10
topic: "Distillation and Adaptation"
level: "intermediate"
tags: ["optimization", "ai-infrastructure"]
heroImage: './section-overview.png'
---

## Overview

![Concept overview: LoRA: Low-Rank Updates and Training-State Memory](./section-overview.png)

Low-rank adaptation changes which parameters are trained. LoRA keeps a pretrained weight matrix fixed and learns an additive update represented by 2 smaller matrices. The central efficiency win is less trainable state. The execution tradeoff depends on whether adapters stay separate or get merged for deployment.

This article derives the matrix geometry, the initialization, the gradients through its 2 factors, and the memory accounting: a low-rank update is a structural constraint on adaptation, not a claim that the pretrained matrix itself has low rank, and that distinction explains both the method's usefulness and the limits of parameter-count comparisons.


*An original conceptual illustration. Numerical plots and examples are illustrative unless explicitly identified as measured evidence.*

## Deep dive

### 1. Define the adapted linear layer

![Deep-dive illustration: Define the adapted linear layer](./deep-dive.png)

Let W_0 be a pretrained matrix mapping an input of width k to an output of width d. LoRA freezes that matrix and represents the learned update with factors A and B of rank dimension r.

$$
W_0\in\mathbb R^{d\times k},\quad A\in\mathbb R^{r\times k},\quad B\in\mathbb R^{d\times r},\qquad W=W_0+sBA.
$$

The scalar s controls the size of the update added to W_0. The original LoRA paper uses alpha divided by r under its stated convention. Implementations and variants can change scaling, so record the actual expression.

The rank of BA is at most r, while W_0 can still be full rank, and so can the adapted matrix W: LoRA constrains the change to the model rather than replacing the whole pretrained mapping with a low-rank approximation.

### 2. Count trainable parameters

Full adaptation of this matrix trains d times k values. LoRA trains r times the sum of d and k, 2 factor matrices instead of 1 full one, excluding any additional trainable biases or modules under the selected recipe.

$$
N_{\mathrm{full}}=dk,\qquad N_{\mathrm{LoRA}}=r(d+k),\qquad \eta=\frac{r(d+k)}{dk}.
$$

For an illustrative square matrix with width 4,096 and rank 16, the full matrix contains 16,777,216 weights. The factors contain 131,072 values, or about 0.781 percent as many trainable parameters for this matrix.

That 0.781 percent does not describe complete training memory, because the frozen base still occupies storage and backward computation can require activations, so count every selected matrix and any trainable embedding, normalization, output head, or bias before reporting the whole-model trainable fraction.

### 3. Interpret low-rank update geometry

The product BA maps the input through an r-dimensional intermediate space, 16 dimensions in the example above, before returning to the d-dimensional output. A selects combinations of input directions, while B maps those combinations into output directions.

This bottleneck limits the update's matrix rank. It does not require the adapted model's useful information to fit into only r dimensions globally, because the pretrained mapping W_0 remains available and several layers can receive independent adapters.

Increasing r expands the possible update space while adding state and computation, and the useful rank depends on the task, chosen layers, data, and optimization, so do not treat a rank of 16 chosen for one checkpoint or domain as a universal prescription for every adaptation problem.

### 4. Preserve the original function initially

The original LoRA setup initializes A randomly and B to zero. So their product is zero at initialization, and the adapted layer starts with the original pretrained output of W_0 under the stated conditions.

$$
B_0=0\quad\Longrightarrow\quad W_0+sB_0A_0=W_0.
$$

This is useful because adaptation starts from a known reference function, W_0 itself. If both factors were initialized to zero, the bilinear parameterization would produce zero gradients for both under the ordinary loss derivative at that point.

Dropout, scaling, and additional trained modules can affect the complete recipe. Verify agreement in the actual inference and training modes rather than assuming the matrix identity makes every surrounding operation equal.

### 5. Derive gradients through the factors

Let G be the derivative of the task loss with respect to the effective matrix W. Matrix calculus gives the gradients of the 2 factors under the chosen orientation.

$$
\frac{\partial\mathcal L}{\partial B}=sGA^\top,\qquad \frac{\partial\mathcal L}{\partial A}=sB^\top G.
$$

At initialization with B equal to zero, so that W equals W_0, the gradient of A is zero while B can receive a nonzero gradient because A is random, and after B changes A can also learn: this asymmetry explains why the initialization avoids the all-zero dead point.

The formulas describe the ideal differentiable parameterization. Actual training includes optimizer state, finite precision, batching, and potentially dropout. A tiny numerical gradient check can verify shapes and scaling, but it says nothing about the quality of a complete adaptation run.

### 6. Explain parameterization nonuniqueness

For any invertible r-by-r matrix R, 16-by-16 in the running example, the factors BR and R inverse A represent the same update. The effective matrix depends on their product, not on a unique set of factor coordinates.

$$
BA=(BR)(R^{-1}A).
$$

Optimization still acts on the factors, so equivalent products need not follow identical training trajectories under different initializations and optimizer settings. Factor scale affects gradient magnitudes and conditioning.

This is one reason rank and alpha do not fully specify a LoRA recipe: initialization, learning rates, regularization, selected modules, and precision all influence training, so compare actual configurations under consistent data and evaluation rather than interpreting factor coordinates as a unique learned explanation.

### 7. Account for optimizer memory

![Deep dive: 7. Account for optimizer memory](./deep-dive-component-03.png)

An optimizer such as Adam commonly keeps moment estimates for each trainable value. Some mixed-precision setups also keep a wider-precision master parameter copy. Freezing base parameters skips their optimizer state, provided the implementation respects the freeze.

$$
M_{\mathrm{trainable}}\approx N_{\mathrm{trainable}}(b_p+b_g+b_{m_1}+b_{m_2}+b_{\mathrm{master}}).
$$

Here each b denotes bytes per value for parameters, gradients, the 2 Adam moments, and any master copy, so omit the master term only when the implementation does not allocate it: this accounting expression is configuration-dependent, not a fixed byte rule for all libraries.

The frozen base W_0, activations, workspace, and framework allocations remain separate terms. Verify that gradients and optimizer slots are absent for frozen tensors. Accidentally including base parameters in the optimizer can undo the intended memory reduction.

### 8. Do not erase activation cost

Training A and B requires signals from layer inputs and backward propagation through the surrounding graph. Freezing W_0 does not mean the entire network can run without storing or recomputing activations.

Long sequences and large batches can make activation memory large even when trainable parameter state is small, the 131,072 values of the earlier example, and activation checkpointing trades stored activations for recomputation at its own execution cost, so LoRA and checkpointing address different memory categories and can be combined deliberately.

Measure peak allocation for the actual sequence length, batch, checkpoint policy, and precision. A trainable-parameter percentage is useful structural information, but it cannot tell you whether a training job fits. Preserve the complete memory model when comparing full adaptation and parameter-efficient alternatives.

### 9. Compute separate adapter execution

For one input vector, the adapter path computes A x followed by B times that intermediate. Its dense multiply-add count is proportional to r times k plus d.

$$
C_{\mathrm{adapter}}\approx2r(k+d),\qquad C_{\mathrm{base}}\approx2dk.
$$

These operation counts omit dispatch, memory traffic, dropout, and additions. Small matrix operations can run inefficiently, so do not present the parameter ratio as a latency ratio.

For batches or sequences, the matrix shapes change and supported fused paths can matter, so measure the actual backend with the intended adapters active, because an adapter that adds little arithmetic can still introduce visible overhead in a small-batch decode workload.

### 10. Merge for a fixed deployment

![Deep dive: 10. Merge for a fixed deployment](./deep-dive-component-01.png)

When the base W_0 is stored in a compatible representation and the deployment uses one fixed adapter, you can materialize the effective weight before inference. This removes the separate low-rank branch from the mathematical linear layer.

$$
W_{\mathrm{merged}}=W_0+sBA.
$$

Real-number equivalence does not imply bitwise equality after finite-precision arithmetic. Compare merged and unmerged outputs under defined tolerances. Quantized bases require additional care because dequantization, addition, and requantization can change the numerical artifact.

A merge also changes operational flexibility, because switching adapters now involves selecting or rebuilding a merged checkpoint, while separate adapters can share one W_0: choose the deployment arrangement from workload and support rather than assuming merging is always the best service architecture.

### 11. Serve several adapters over one base

A shared base W_0 plus separate adapter sets can reduce duplicated storage across tasks. If each adapter contains N_a values, K adapters add roughly K times their parameter bytes rather than K full copies of the base, before runtime overhead.

Serving multiple adapters raises scheduling and kernel questions. Requests can select different updates, and batching across adapters may require specialized execution or grouping. The resulting throughput depends on the workload's adapter distribution and backend support.

Report whether memory includes all resident adapters and whether the comparison batches compatible requests, since a single-adapter benchmark does not tell you how a service behaves when it switches among many adapters: shared storage and efficient mixed-adapter execution are related but separate engineering results.

### 12. Select modules and rank with evidence

Adapting attention projections, feed-forward matrices, or another subset changes the available update space. A small adapter placed in the right modules can outperform a larger but poorly chosen arrangement for a particular task.

Use a transparent baseline and vary module selection and rank under a controlled data budget. Track quality, trainable state, preparation time, and exported execution. Rank alone is not enough to compare on when the number and shapes of adapted matrices differ.

Also inspect retained trainable modules outside LoRA. Training an output head or embeddings can materially change the parameter and memory totals. State those choices alongside the factor ranks so the reported efficiency is reproducible.

### 13. Compare against full adaptation fairly

![Deep dive: 13. Compare against full adaptation fairly](./deep-dive-component-04.png)

Full adaptation and LoRA optimize different parameter spaces, 16,777,216 weights against 131,072 in the earlier example, so their learning-rate schedules, regularization, and suitable training budgets may differ, and a comparison should document these choices rather than forcing an arbitrary identical recipe and declaring one method universally superior.

Evaluate on held-out tasks with the same preprocessing and generation policy. Include the pretrained reference so that readers can see the actual adaptation gain. Check whether domain gains come with regressions elsewhere when that matters to deployment.

No model adaptation or GPU timing was performed for this article. The 4,096-wide parameter example and the compute formulas are illustrative accounting. The original paper provides experiments under its configurations; a new task needs evidence from its own checkpoint, data, and backend.

### 14. Connect LoRA to statistical constraints

Restricting updates to a low-rank product can act as an inductive constraint on adaptation. It reduces the number of free trainable values and may be useful when the available task data does not justify unconstrained changes to every base weight.

That interpretation does not guarantee improved generalization. A low rank can underfit, while several adapters across many layers can still provide substantial capacity. Regularization, data quality, and task structure remain important.

LoRA's innovation is a concrete factorized update, 2 matrices in place of 1, that preserves the pretrained mapping while reducing trainable state, and its deployment benefit depends on memory accounting and merge or adapter execution, so keep those mechanisms distinct from a broad claim that every low-rank training recipe is automatically cheaper in every phase.

### 15. Verify the numerical and export contract

![Deep dive: 15. Verify the numerical and export contract](./deep-dive-component-02.png)

Run a small diagnostic that compares the direct effective matrix with the separate adapter branch on nontrivial inputs. Include a nonunit scale so that a missing alpha-over-r factor is visible. Verify that the initial update leaves W_0 unchanged and check factor gradients against finite differences in a small wider-precision example.

The exporter must preserve the selected modules, factor orientation, scale, and any trained bias. A checkpoint containing A and B without the identity of its W_0 base is not a self-contained description of the adapted function. Store the exact base revision and preprocessing settings with the adapter artifact.

## Conclusion

For deployment, compare merged and separate outputs under the intended numerical policy, check that training-only operations are disabled, and then measure the complete inference path rather than relying on the adapter's count of about 2r times k plus d: these checks connect the low-rank equation to a usable artifact without confusing algebraic equivalence with implementation correctness or measured speed.

### Sources

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314).
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353).
