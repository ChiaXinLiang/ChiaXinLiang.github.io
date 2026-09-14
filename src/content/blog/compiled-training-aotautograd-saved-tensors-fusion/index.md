---
title: "Compiled Training: AOTAutograd, Saved Tensors, and Fusion Boundaries"
description: "Follow forward and backward graph capture, derive saved-tensor and recomputation tradeoffs, and validate compiled training beyond inference timing."
pubDate: "Sep 13 2026"
updatedDate: "Sep 13 2026"
heroImage: './section-overview.png'
series: "gpu-performance"
code: "pt-3"
order: 25
topic: "PyTorch and Compilers"
level: "advanced"
tags: ["gpu-performance", "ai-infrastructure"]
---

## Overview

![Concept overview: Compiled Training: AOTAutograd, Saved Tensors, and Fusion Boundaries. A forward computation graph and backward graph share saved activation tensors.](./section-overview.png)

Compiling a training program is more involved than compiling its forward pass, because the program carries 4 obligations: produce a valid loss, compute gradients with the intended differentiation rules, preserve state updates, and respect tensor lifetimes across forward and backward execution. A fast inference graph can still leave the expensive training path unchanged or introduce a costly saved-tensor boundary.

AOTAutograd provides a way to trace differentiable computation and expose forward and backward graphs to compilation. The name refers to preparing those graphs before their compiled execution, not to a promise that the entire application has become 1 permanent ahead-of-time binary. Actual integration, graph capture, and supported behavior depend on the PyTorch version and backend.

## Deep dive

### 1. Write the differentiation contract

![Deep-dive illustration: Write the differentiation contract](./deep-dive.png)

Consider a function producing a scalar loss from parameters and input data: the training step needs 2 results, the loss and its derivatives with respect to the intended parameter population, and frozen parameters, detached tensors, and nondifferentiable operations all alter that population. The compiler must preserve those semantics rather than differentiating every available tensor.

$$
\ell=f(\theta,x),\qquad g=\nabla_{\theta}f(\theta,x),\qquad
\theta_{t+1}=U(\theta_t,g_t,s_t).
$$

The update function also depends on optimizer state s, and 4 mechanisms make that state meaningful: momentum, adaptive moments, gradient accumulation, and mixed-precision scaling. A forward-only comparison does not test the complete update function.

Establish which operations belong to the compiled region. The loss can be compiled while the optimizer remains outside it, or a larger step can be captured under supported conditions. State the actual boundary in the benchmark. “Compiled training” is otherwise too broad to interpret.

### 2. Trace the joint computation

A differentiation system can expose a joint representation of forward operations and the operations needed to calculate gradients. AOTAutograd tracing makes that computation available for partitioning and backend compilation. The resulting representations need not resemble the handwritten Python function line by line.

Tracing observes tensor operations under particular assumptions about inputs, shapes, state, and supported behavior, and Python-side effects or unsupported control flow can remain outside captured regions or cause capture difficulties. Dynamic shapes and guards are therefore relevant to training as well as inference.

The joint graph gives a compiler visibility into relationships that separate forward and backward optimization would miss, and it cuts 2 ways: a cheap forward intermediate might be recomputed in backward instead of saved, while an expensive operation may justify retaining its result even if that increases memory use.

### 3. Derive a saved-tensor decision

![Deep dive: 3. Derive a saved-tensor decision](./deep-dive-component-01.png)

Suppose a backward operation needs an intermediate z produced by an earlier forward operation, which leaves 2 choices: saving z consumes memory over its live interval, while recomputing z avoids that saved allocation but repeats work and may require saving some of its inputs. The comparison must include the entire dependency closure, not just z's own size.

A simplified objective combines execution time with a penalty for memory pressure:

$$
J=T_{\mathrm{forward}}+T_{\mathrm{backward}}+T_{\mathrm{recompute}}+\lambda M_{\mathrm{peak}}.
$$

The coefficient is a modeling device, not a universal compiler setting. Real systems have a hard memory-capacity constraint and discrete kernel choices. The equation nevertheless clarifies why minimizing saved bytes alone can select an unacceptably slow schedule.

For an illustrative intermediate of 16 megabytes whose producer is inexpensive element-wise arithmetic, recomputation may be attractive if its inputs already remain live. If producing the same-size intermediate requires a large matrix multiplication and retaining its inputs adds substantial memory, the tradeoff changes. Equal output size does not imply equal recomputation cost.

### 4. Understand graph partitioning

A partitioner divides the joint graph into 2 execution regions, forward and backward, and determines which values cross their boundary: saved tensors are part of that interface, and their placement affects memory lifetime, transfer cost where relevant, and opportunities to fuse surrounding operations.

The official AOTAutograd optimization tutorial discusses partitioning strategies, including approaches that use graph structure to trade recomputation against saved values. Treat its examples as explanations of the method. The exact partitioner and configuration in a current torch.compile backend can differ from a standalone historical tutorial.

A useful inspection asks 3 questions: which values are outputs of the compiled forward graph for backward use, which backward operations recompute values, and which inputs must survive. Do not infer those decisions from the source code alone, because generated graphs and memory observations provide the relevant evidence.

### 5. Work through a small derivative

Let an element-wise forward computation take 3 steps: multiply a parameter by an input, apply a sigmoid, and sum the resulting values. The backward gradient depends on the sigmoid output and the input, and it need not save the product if the chosen differentiation path has sufficient other values.

$$
z_i=\theta_i x_i,\quad y_i=\sigma(z_i),\quad
\ell=\sum_i y_i,\quad
\frac{\partial\ell}{\partial\theta_i}=x_i y_i(1-y_i).
$$

1 partition can save y and x, while another can retain the required inputs and recompute z and y during backward, and both implement the same real-number derivative, though their floating-point behavior and memory traffic can differ. If x is already live for another reason, the incremental storage cost differs from a case in which retaining x extends its lifetime.

This example explains the boundary without suggesting that the compiler always selects 1 policy. Measure the actual graphs. A numerical test should compare the gradient as well as the scalar loss, using a tolerance appropriate to precision and expected operation reordering.

### 6. Explain fusion boundaries

Fusion combines compatible operations into fewer kernels, can avoid writing intermediate tensors to device memory, and can reduce launch overhead, while compatibility depends on 5 properties: iteration domains, layouts, reduction structure, numerical behavior, and backend capabilities.

A saved value can force an intermediate to remain available after the forward region finishes, so an otherwise fusible expression may still need an output write for backward, and a partition that recomputes that expression changes the materialization requirement at the cost of extra backward work. This is 1 reason training fusion cannot be assessed from forward code alone.

Larger fused regions can increase register pressure and live state. They may constrain scheduling around reductions or matrix operations. A kernel-count reduction is evidence about launches, not a sufficient performance result. Record elapsed time and resource use for the complete relevant step.

### 7. Preserve mutations and aliasing

Training code often changes parameters, optimizer buffers, counters, or accumulated gradients. Tensor views can share storage even when their Python objects differ. A compiler must represent those dependencies correctly, including mutations that become observable outside the captured region.

Functionalization can transform supported mutations and view behavior into a representation more suitable for graph transformations. That does not grant permission to alter the program's observable state. Check the resulting values and alias-sensitive behavior at the actual interface.

An in-place operation can also violate autograd's own saved-value requirements independently of compilation. Establish a correct eager baseline first. A compiler failure caused by an invalid differentiation program should not be addressed by silently changing the mathematical training objective.

### 8. Include randomness and precision

Dropout and other stochastic operations influence both forward values and gradients, and recomputing them requires a valid relationship to the intended random-number state. Examine 3 mechanisms under their documented behavior rather than assuming they redraw equivalent masks: activation checkpointing, graph partitioning, and backend lowering.

Mixed precision introduces 4 concerns: cast placement, accumulator precision, loss scaling, and possible skipped updates, so a compiled step must preserve the intended handling of nonfinite gradients and scaling state. Comparing only finite forward outputs can miss an update-semantic mismatch.

Control random seeds and initial state for comparisons, but do not assume every compiled implementation reproduces the eager random sequence bit for bit. State whether the test requires exact reproducibility, numerical closeness under matched randomness, or a statistical property. Those are 3 different validation contracts.

### 9. Measure compilation separately

The first execution can include tracing, graph transformation, code generation, autotuning, and initialization. Report cold-start cost separately from warmed steady-state step time. Recompilation under changed shapes or guards belongs in a workload measurement when the real input distribution triggers it.

Use a consistent timing boundary covering forward, backward, and whichever optimizer operations the claim includes. Device synchronization or events must establish the completion boundary. Host enqueue time alone does not measure GPU training duration.

Record peak allocated memory and, when useful, reserved memory separately. The allocator's reserved pool is not the same as simultaneously live tensors. A reduction in saved tensors can change peak allocation without immediately reducing reserved capacity. Explain the measured quantity instead of presenting 1 memory number as universal.

### 10. Validate gradients and trajectories

![Deep dive: 10. Validate gradients and trajectories](./deep-dive-component-03.png)

Compare eager and compiled runs from identical parameters and optimizer state. Test loss, selected intermediate outputs, gradients, and updated parameters. Exercise accumulation, zeroing behavior, supported shapes, and noncontiguous layouts if they belong to the input contract.

For a small deterministic function, finite differences can independently check selected derivatives:

$$
\frac{\partial f}{\partial\theta_i}\approx
\frac{f(\theta+\epsilon e_i)-f(\theta-\epsilon e_i)}{2\epsilon}.
$$

Choose the perturbation with floating-point error in mind; making it arbitrarily small increases cancellation problems. Finite differences on a small smooth case do not validate every large-model training path, but they can expose a mistaken gradient reference.

A few successful steps establish local agreement. A longer trajectory can reveal accumulating state or precision differences, although exact trajectories may diverge under allowed floating-point reordering. Define acceptable evidence for the use case and keep the eager reference reproducible.

### 11. Build a useful regression record

Store the PyTorch version, backend, compile options, shape distribution, dtype, initial state, and timing method. Include graph breaks and recompilation behavior when they materially affect the result. A performance regression can originate in partitioning, generated kernels, saved-value lifetime, or a changed workload boundary.

Compare 1 change at a time when diagnosing. If peak memory rises, inspect new saved values and their live intervals before assuming an allocator bug. If backward becomes slower, inspect recomputed work and fusion boundaries. If only cold-start worsens, separate compilation from steady-state execution rather than calling both the same regression.

The methods and derivative examples here have not been benchmarked on a GPU in this editing environment. They explain how to assess compiled training: preserve the differentiation and update contract, inspect the forward-backward boundary, and measure the complete supported step.

### 12. Distinguish live intervals from tensor totals

![Deep dive: 12. Distinguish live intervals from tensor totals](./deep-dive-component-02.png)

Suppose the forward pass creates 3 intermediate tensors, each occupying 10 megabytes. Adding their sizes gives 30 megabytes, but that is not necessarily the peak: if the first tensor dies before the third is created, only 2 may coexist, whereas saving the first tensor for backward can extend its lifetime until all 3 coexist. The saved-value decision changes the overlap of intervals rather than merely the number of tensors created.

Backward execution can create its own temporaries while saved forward values are still live, so the actual peak can occur after the forward pass has finished, which means you should measure the complete step and inspect where the high-water mark falls. A forward-only allocation trace can understate the training requirement even when every forward intermediate is accounted for.

Recomputation has a similar subtlety. Recomputing a 10-megabyte tensor avoids its long saved interval, but briefly materializing it beside a large backward temporary can still contribute to peak memory. Fusion may avoid that materialization, whereas a backend boundary may preserve it. The partition and lowering decisions interact, which is why a simple sum of saved tensor sizes is only a diagnostic quantity.

## Conclusion

For a concrete review, draw each important value's lifetime from creation to last use on a forward-backward timeline. Mark the tensors crossing the partition boundary and the temporaries introduced by backward. Then compare the timeline with measured peak allocation. An unexplained discrepancy identifies missing state, aliasing, allocator behavior, or an incorrect assumption about generated execution. This exercise makes a memory claim reviewable without pretending the source-level graph determines every runtime allocation.

### Sources

- [Official AOTAutograd optimization tutorial](https://docs.pytorch.org/functorch/stable/notebooks/aot_autograd_optimizations.html).
- [PyTorch torch.compile tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).
- [PyTorch dynamic shapes documentation](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html).
