---
title: "Distillation 2: Features, Data, and Deployment Tradeoffs"
description: "Matching teacher probabilities is one way to distill a model."
pubDate: "2026-09-13"
updatedDate: "2026-09-13"
series: "efficient-ml"
code: "eml-9"
order: 9
topic: "Distillation and Adaptation"
level: "intermediate"
tags: ["optimization", "ai-infrastructure"]
heroImage: "./cover.png"
---

Matching teacher probabilities is one way to distill a model. Another transfers intermediate representations, relations between examples, or complete generated sequences. These objectives expose different information and create different preparation costs. The central design question is what a constrained student should reproduce to improve the task that matters.

Feature matching does not require the student to become an exact internal copy. Different architectures can represent the same decision using different coordinates. A useful distillation interface therefore specifies alignment, invariances, data, and evaluation. This article connects those theoretical choices to an efficient deployable student.

![Concept overview: Distillation 2: Features, Data, and Deployment Tradeoffs](./section-overview.svg)

*An original conceptual illustration. Numerical plots and examples are illustrative unless explicitly identified as measured evidence.*

## 1. Define an intermediate interface

Let h_t(x) be a teacher representation and h_s(x) the student representation for input x. Their dimensions and spatial or sequence resolutions can differ. Introduce an alignment map P when direct comparison is not meaningful.

$$
h_t(x)\in\mathbb R^{d_t},\quad h_s(x)\in\mathbb R^{d_s},\qquad P:\mathbb R^{d_s}\rightarrow\mathbb R^{d_t}.
$$

FitNets introduced intermediate hints and a mapping to support thinner students. The broader mechanism is to provide supervision inside the model rather than only at its final predictions.

Choose the layer correspondence deliberately. Equal layer indices do not imply equal semantic depth when teacher and student architectures differ. A mapping is part of the method and needs documented shapes, initialization, and training status.

## 2. Write a feature reconstruction objective

A simple aligned feature loss uses squared distance, normalized here by teacher feature width. Its gradient teaches both the student representation and any trainable projector.

$$
\mathcal L_{\mathrm{feat}}=\frac1{d_t}\|P h_s(x)-h_t(x)\|_2^2.
$$

The normalization makes the convention explicit; implementations can use another reduction. The loss scale affects how it combines with task supervision, so record whether reductions average across dimensions, examples, or positions.

A small feature error is a local achievement. It does not guarantee equal final outputs, and it can be dominated by high-magnitude coordinates. Inspect the feature distribution and downstream task rather than interpreting the reconstruction number as a complete quality measure.

## 3. Recognize representation nonuniqueness

Suppose a hidden representation is transformed by an invertible matrix R and the next linear layer is transformed by its inverse. The composed real-number function can remain unchanged even though the hidden coordinates differ.

$$
W h=(W R^{-1})(R h).
$$

This demonstrates why raw feature equality is stronger than functional equivalence. Two useful models can encode similar information in rotated or rescaled coordinate systems. An alignment projector can absorb some of those differences, but its capacity and constraints matter.

Conversely, a very powerful projector can fit teacher features while leaving the student poorly prepared for the task. Feature matching should support learning rather than become an auxiliary network that solves the comparison independently of useful student behavior.

## 4. Compare normalized features

Normalizing representations focuses a loss on direction rather than unrestricted magnitude. A small positive epsilon stabilizes the denominator under the chosen implementation.

$$
\widetilde h=\frac{h}{\|h\|_2+\varepsilon},\qquad \mathcal L_{\mathrm{direction}}=\|\widetilde {P h_s}-\widetilde h_t\|_2^2.
$$

This changes the objective. Magnitude can carry useful confidence or activation information, so removing it is an invariance choice rather than an automatic improvement. Near-zero vectors also require a defined policy.

Evaluate normalized and unnormalized objectives under the same student and data budget when the choice is uncertain. Inspect whether one objective makes the representation comparison easier while weakening downstream quality. A convenient similarity score should not substitute for the application objective.

## 5. Transfer relations between examples

Relational knowledge distillation compares structure across a set of representations. Its primary paper develops distance-wise and angle-wise objectives. The general idea is to preserve how examples relate rather than require every feature coordinate to match.

For example, define pairwise distances within a minibatch and normalize them by a nonzero mean distance. Teacher and student can then compare relative geometry despite differing overall scales.

$$
d_{ij}(h)=\|h(x_i)-h(x_j)\|_2,\qquad \bar d_{ij}=\frac{d_{ij}}{\operatorname{mean}_{a\ne b}d_{ab}}.
$$

This illustrative definition needs safeguards for degenerate batches. It also makes batch composition part of the supervision. A batch containing only nearly identical examples supplies different relational information from one spanning several classes or behaviors.

## 6. Work through geometric invariance

Take 3 illustrative teacher points: the origin, a point one unit along the first axis, and a point one unit along the second. Translate and rotate all 3 points to produce student points. Pairwise distances remain identical while raw coordinate differences can be large.

A distance-based objective therefore treats these configurations as equivalent. Uniform scaling also disappears after the chosen distance normalization. This can be useful when architectures use different feature coordinates.

The same invariance can hide distinctions an application needs. Pairwise distance preservation alone does not require identical classification heads or confidence. Combine appropriate task supervision and evaluate the final student. The geometric example explains what the loss ignores as well as what it preserves.

## 7. Budget the alignment computation

A dense linear projector from width d_s to d_t contains d_s times d_t weights, plus a bias if used. Several intermediate matches can add substantial training computation and activation storage.

$$
N_P=d_s d_t,\qquad C_P\approx2N d_s d_t
$$

The compute estimate uses N compared representations and counts a multiply-add as 2 floating-point operations. It excludes backward work and memory traffic. Actual preparation cost also depends on the teacher and how features are captured.

A projector used only for training can be removed from the deployed student. Verify that removal in the exported graph. If inference still requires the teacher or alignment network, the deployment claim must include those components rather than counting only the student backbone.

## 8. Distill complete sequences

Sequence-level distillation uses teacher-produced outputs as training targets. For an autoregressive student, a selected teacher sequence y_star can define an ordinary conditional negative log-likelihood objective.

$$
\mathcal L_{\mathrm{seq}}=-\sum_{u=1}^{|y_*|}\log p_s(y_{*,u}\mid x,y_{*,<u}).
$$

This differs from comparing every teacher token distribution. The selected sequence collapses a distribution of possible outputs into a particular target, with consequences for diversity and ambiguity.

The original sequence-level work studied sequence-to-sequence learning. Applying the same broad mechanism elsewhere requires checking task and generation policy. A beam-selected translation, a sampled response, and a filtered reasoning trace represent different supervision populations and should retain that provenance.

## 9. Understand target-selection bias

Teacher generation depends on prompts, decoding, length limits, sampling, and filtering. Those choices determine what the student sees. A target set can overrepresent polished short answers or omit difficult cases rejected by a quality filter.

Data filtering can be useful, but it changes the training distribution. Record rejection rates and inspect which populations disappear. A student trained only on teacher successes has not learned how to recover from every failure it may encounter during deployment.

Also distinguish teacher correctness from target fluency. Fluent generated text can contain errors. Evaluate targets using task-appropriate evidence when available, and preserve independent labels or references for final evaluation. The student can reproduce a consistent error pattern with high likelihood.

## 10. Model the data-generation cost

Let N be the number of prompts, c_t their average teacher-generation cost, and c_v the average verification cost. A simple preparation estimate adds those costs to student training.

$$
C_{\mathrm{prep}}\approx N(c_t+c_v)+C_{\mathrm{student\ training}}.
$$

The units must agree: device time, monetary cost, energy, or another chosen resource. Include retries, discarded targets, and storage when they matter. Counting only accepted outputs understates generation effort when filtering is aggressive.

The deployment savings must justify this preparation under the expected reuse. A frequently deployed student can amortize a large teacher-generation phase, while a one-off task may not. Report preparation and inference separately so that readers can apply their own reuse assumptions.

## 11. Control task and data comparisons

Compare distillation variants on the same student architecture and evaluation setup. If one variant uses substantially more generated data, its improvement cannot be attributed solely to feature or sequence supervision.

Useful ablations separate hard targets, soft probabilities, feature losses, and relational losses when those components are combined. Keep training budgets explicit. Equal epoch counts do not imply equal processed examples or equal teacher computation.

Select methods using a validation population and reserve independent evaluation for the final artifact. Repeated selection against one test benchmark can overfit the distillation recipe even when student weights never directly train on those benchmark labels.

## 12. Inspect capacity mismatch

A teacher can learn distinctions that a smaller student cannot represent under its architecture. Increasing the feature-loss weight can then force an unfavorable compromise with the task objective.

Inspect whether training reduces auxiliary feature error while task quality stagnates or declines. That pattern can indicate a poor layer correspondence, insufficient projector design, incompatible capacity, or an inappropriate supervision population. It should lead to a concrete diagnostic rather than an unexplained larger loss weight.

Intermediate supervision can also make optimization easier without requiring exact teacher representation. Select interfaces that preserve task-relevant information and allow the student its own efficient structure. An architecture-specific mapping should be assessed for what it teaches and what it costs.

## 13. Check inference independence

The final student should have a clearly defined inference graph. Confirm which training-only modules were removed, which parameters were retained, and whether preprocessing still depends on a teacher-derived service.

Measure the exported artifact on the intended backend. Parameter reduction can coexist with unfavorable matrix shapes or extra activation traffic. Report latency and throughput under a documented workload, alongside memory and task quality.

No student training or GPU execution was performed for this article. The equations and geometric examples are explanatory. The primary papers provide their own experimental evidence under their stated tasks, while a new deployment requires measurements of its actual student and numerical policy.

## 14. Choose the interface from the objective

Use output probabilities when relative alternatives provide useful supervision and the class interface is compatible. Use intermediate hints when a justified alignment can improve representation learning. Use relational objectives when preserving geometry is more meaningful than matching coordinates. Use generated sequences when the task benefits from selected teacher outputs and their distribution is understood.

These options can be combined, but each component adds assumptions and preparation work. Begin with a transparent baseline and add supervision that addresses an observed limitation.

The most useful explanation of distillation identifies the information transferred, the invariances imposed, the data that carries it, and the student execution that follows. That connects statistical learning to efficiency without treating teacher imitation as an automatic guarantee of quality or speed.

## 15. Examine batch effects in relational learning

Pairwise objectives can involve a quadratic number of pairs in the minibatch, while angle-based comparisons can involve more expensive tuple construction. Practical methods sample or organize these relations to control preparation cost. The selected relation population should be documented because it determines which geometry the student is encouraged to preserve.

A batch with repeated or nearly identical examples can make distance normalization unstable or uninformative. A batch composed of unrelated examples can emphasize broad separation while omitting fine distinctions within a class. These are statistical properties of the supervision rather than merely implementation details.

For a diagnostic, construct a small batch with known distances, verify the normalization and zero-distance policy, and compare the loss after translation, rotation, and scaling. Then inspect batches sampled from the actual training data. The synthetic geometry tests correctness; the real population tests whether the objective carries useful information.

Avoid interpreting a low relational loss as an assurance that the student preserves every teacher behavior. The loss only constrains selected relations under selected inputs. Independent task evaluation remains necessary, especially when the deployed distribution differs from the data used to create teacher features.

## Sources

- [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550).
- [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068).
- [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947).
- [Contrastive Representation Distillation](https://arxiv.org/abs/1910.10699).
