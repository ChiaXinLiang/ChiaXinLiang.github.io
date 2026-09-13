---
title: "Pruning 1: Magnitude, Saliency, and Recovery Training"
description: "Pruning removes selected connections or components from a learned network."
pubDate: "2026-09-13"
updatedDate: "2026-09-13"
series: "efficient-ml"
code: "eml-2"
order: 2
topic: "Pruning and Sparsity"
level: "intermediate"
tags: ["optimization", "ai-infrastructure"]
heroImage: "./cover.png"
---

Pruning removes selected connections or components from a learned network. Its appeal is straightforward: a model can contain more parameters than a deployment needs for its task. The difficult questions are which parameters to remove, how to recover useful behavior, and whether the resulting representation actually executes more efficiently.

This article treats pruning as a constrained change to a trained function. It develops magnitude and loss-based importance criteria, explains their assumptions, and follows the recovery experiment. The next article addresses structured patterns and hardware execution. Separating those questions prevents a sparse checkpoint from being mistaken for a demonstrated speedup.

![Concept overview: Pruning 1: Magnitude, Saliency, and Recovery Training](./section-overview.svg)

*An original conceptual illustration. Numerical plots and examples are illustrative unless explicitly identified as measured evidence.*

## 1. Represent the intervention with a mask

Let theta contain trained parameters and let m be a binary mask with the same logical shape. The effective parameters are their element-wise product. An illustrative pruning objective minimizes task loss under a nonzero budget:

$$
\min_{m,\theta'}\mathcal L(m\odot\theta')\quad\text{subject to}\quad
m_i\in\{0,1\},\ \|m\|_0\le K.
$$

The zero-norm notation counts retained entries rather than defining an ordinary vector norm. The problem combines discrete structure selection with possible weight recovery. Solving it exactly is generally impractical for a large network, motivating importance heuristics and approximate optimization.

A mask can remain applied to a dense tensor during experimentation. That establishes sparse semantics but not compressed storage or skipped arithmetic. Deployment requires a compatible representation and kernel. Keep the mathematical intervention separate from its execution.



![Deep-dive illustration: Represent the intervention with a mask](./deep-dive.png)

## 2. Explain magnitude pruning

Magnitude pruning retains parameters with larger absolute values under a chosen population. A threshold creates the mask; alternatively, a top-K rule retains a specified number. The original learning-connections paper uses training, low-weight pruning, and retraining as a complete procedure.

$$
m_i=\mathbf 1\{|\theta_i|\ge\tau\}.
$$

Small values can have small immediate effects in some settings, making magnitude a cheap useful signal. It is not a universal measure of importance. Input scale, normalization, parameterization, and downstream amplification affect the contribution of a weight.

Define whether the ranking is global, per layer, or per component. A global threshold can remove disproportionate structure from a layer whose numerical scale differs from others. A per-layer budget protects against that particular imbalance but can retain unnecessary parameters in insensitive layers. Evaluate the allocation policy rather than treating its choice as harmless.

## 3. Use a Taylor approximation to loss

A loss-based saliency asks how removing a parameter changes the objective. If parameter i is set to zero, its perturbation is minus its current value. A local second-order expansion gives:

$$
\Delta\mathcal L_i\approx-g_i\theta_i+\frac12H_{ii}\theta_i^2.
$$

Here g is the loss gradient and H its Hessian at the evaluated parameters. Near a stationary point, the gradient term can be small, making diagonal curvature relevant. During active training or under distribution shift, that simplification need not hold.

The expansion is local, while setting a parameter to zero can be a large perturbation. Cross-parameter interactions are also omitted when scores are evaluated independently. Saliency therefore predicts a change under assumptions rather than proving that a selected sparsity level preserves quality.

Computing exact curvature is expensive. Approximate diagonals, gradient statistics, or layer reconstruction objectives reduce that cost but introduce another modeling choice. Document which statistic is estimated and on what data.

## 4. Derive compensation with coupled curvature

Removing one weight can be partly compensated by changing others. Under a quadratic loss model with positive-definite curvature, minimize perturbation cost while enforcing that the chosen coordinate becomes zero. A constrained solution uses a column of the inverse Hessian:

$$
\delta=-\frac{\theta_i}{(H^{-1})_{ii}}H^{-1}e_i,\qquad
\Delta\mathcal L_{\mathrm{quadratic}}=\frac{\theta_i^2}{2(H^{-1})_{ii}}.
$$

The unit vector e_i selects the coordinate. The perturbation's i-th entry equals minus theta_i, satisfying the constraint. Other entries distribute compensating changes according to coupled curvature. The formula explains why independently zeroing weights can differ from an error-compensated method.

Singular or indefinite curvature needs additional treatment. A practical approximation can add damping or use a reconstruction objective with more convenient structure. This equation is a derivation of the quadratic model, not a claim that every pruning method computes a full model Hessian.

## 5. Work through an importance reversal

Consider 2 scalar weights with magnitudes 0.1 and 1.0. Suppose the first coordinate has diagonal curvature 1,000 and the second has curvature 1 under the local stationary approximation. Their estimated removal costs are 5 and 0.5 respectively.

Magnitude ranking would remove the smaller weight first, while this curvature model prefers removing the larger one. The example is hypothetical and does not establish that curvature always wins. It shows that weight magnitude and sensitivity can disagree because the surrounding function amplifies perturbations differently.

Now introduce a nonzero gradient. The first-order term can change both rankings again. A saliency measured on one batch can also be noisy. Estimate it on representative data and test the actual pruned model. The approximation should help select an intervention, while validation establishes the resulting behavior.

## 6. Examine rescaling invariance

In a compatible positively homogeneous network, one layer's weights can be scaled by a positive factor while the next layer's weights are inversely scaled, preserving the overall function. Magnitudes then change even though the model's predictions do not.

For an illustrative ReLU pair, scaling the first linear map by c and the second by one divided by c preserves the composed function under the stated bias and activation conditions. A naive global magnitude ranking can nevertheless change. This is a parameterization problem rather than evidence that every magnitude method is useless.

Normalization, biases, and architectural details can alter the simple invariance. Use the example to question an importance statistic's assumptions, not to claim all networks have identical symmetries. Layer-aware or activation-aware methods can address some issues, but require their own evaluation.

## 7. Choose the pruning schedule

One-shot pruning removes a selected fraction at once. Iterative pruning alternates smaller removals with recovery. Gradual pruning changes the budget during training. Their cost, quality, and final structure can differ.

Large one-shot perturbations can exceed the regime in which local saliency is informative. Iteration can update importance after the function changes, but it spends more training or calibration work. The appropriate schedule depends on task tolerance and available recovery resources.

Keep the total recovery budget in comparisons. A method receiving substantially more retraining is not a controlled comparison of its selection rule alone. Record removed fraction, iterations, data, optimizer, and the final execution representation. A sparse result should be reproducible as an artifact rather than reconstructed from an ambiguous description.

## 8. Prevent unintended regrowth

Applying a mask in the forward computation does not automatically describe optimizer behavior. Momentum, weight decay, and stored gradients can affect masked parameters unless the implementation handles them consistently. Define whether zeros remain fixed or whether regrowth is part of the method.

For fixed-mask recovery, verify the effective weights stay zero and the intended parameter population receives updates. Optimizer state may still occupy memory for masked dense entries. That matters to training-memory claims even when inference semantics are sparse.

A deployment artifact can physically remove structure or store only retained values. Such conversion should preserve the learned function under the supported sparse semantics. Compare the converted artifact with the masked reference before measuring speed.

## 9. Distinguish calibration and recovery data

Calibration data estimates importance or reconstruction error. Recovery data updates remaining parameters. Held-out evaluation estimates task behavior after the choices are made. Reusing the final test set to tune sparsity weakens the reported quality evidence.

Data distribution matters. A mask chosen on common examples can damage a rare but important subgroup. Evaluate task slices and shift scenarios appropriate to the application. A small average loss increase does not guarantee every required behavior survives.

Use reproducible sampling and retain preprocessing settings. If importance estimates vary substantially across calibration samples, report that variation or choose a more stable procedure. The mask is a learned decision conditioned on data, not an intrinsic truth about which connections are unnecessary everywhere.

## 10. Compare sparsity with actual quality

Sweep a small set of budgets and report task metrics with uncertainty. Compare against the original checkpoint under the same input and inference settings. Include recovery cost and any changed numerical representation.

A useful curve plots quality against retained fraction. A deployment curve should additionally plot measured latency or memory, since retained fraction may not translate directly. These curves can reveal an insensitive region followed by sharp degradation, but the location is task- and model-dependent.

Do not infer acceptable sparsity from a result on another architecture. Layer widths, normalization, routing, and training history change sensitivity. Use primary-paper experiments within their stated setup and measure the actual intended checkpoint.

## 11. Separate pruning from training a smaller model

A pruned trained model and a newly trained smaller model are different procedures. They can have similar parameter counts but different structures, initialization, and training cost. Comparing them can be useful if the complete resource budget is explicit.

Recovery preserves some learned weights, while training from scratch spends another optimization path. Distillation adds teacher supervision. None should be silently included in a pruning speed or quality claim without identifying its contribution.

For an infrastructure decision, compare feasible artifacts and their acquisition cost. An expensive compression procedure can still be worthwhile for many repeated inferences, while a rarely used model may not amortize that work. The experiment should connect preparation cost with deployment frequency.

## 12. Test the mask and conversion

Give a small matrix distinguishable entries, apply a known mask, and compare outputs with an explicit zeroed reference. Verify retained counts, axis conventions, and supported bias behavior. Test mask application after serialization and loading.

For fixed-mask recovery, check that masked contributions remain zero through optimizer steps. For physical conversion, validate the output association and any surrounding dimension changes. Structural tests complement task evaluation; neither replaces the other.

No model-weight execution or GPU benchmark was performed for this article. Its numerical examples illustrate saliency assumptions. The practical sequence is to define a budget, select structure with a documented statistic, recover within an explicit training budget, validate quality, and then measure the deployed sparse representation.

## 13. Relate layer reconstruction to task loss

A scalable pruning procedure can minimize the difference between a layer's original and modified outputs on calibration activations. For linear output WX, the reconstruction objective is the squared norm of the difference between WX and the pruned matrix applied to X. This gives a convenient quadratic structure and avoids calculating curvature for the complete language-model task.

The convenience is also a limitation. Matching one layer's calibration outputs is a surrogate for preserving the final task behavior. Later nonlinearities, residual interactions, and changes in the activation distribution after earlier layers are pruned can affect that relationship. A sequence of locally good approximations can still accumulate error.

A practical procedure should state whether it recalibrates later layers using the already modified network or keeps original activations. Those choices change which distribution the reconstruction objective sees. Neither should be treated as a trivial implementation detail when comparing results.

For a tiny example, retain the original activations and explicitly calculate the reconstruction error before and after compensation. Then compare the complete model's output under the same inputs. This separates a mistaken quadratic solver from a limitation of the surrogate objective. Use held-out task evaluation after selecting the policy, since the calibration objective itself cannot establish final quality.

This connection explains why second-order one-shot methods can be computationally practical while still needing behavioral evidence. Their innovation lies partly in choosing a tractable local problem and solving it efficiently. The deployment review must preserve both the mathematical approximation and its measured consequences.

## Sources

- [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626).
- [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774).
