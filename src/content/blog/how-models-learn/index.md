---
title: 'How Models Learn: Gradient Descent and Backprop in Plain Words'
description: "Training a neural network is finding the bottom of a valley you can't see, 1 step at a time, and billing every weight for its exact share of every mistake."
pubDate: 'Sep 12 2026'
updatedDate: 'Sep 12 2026'
heroImage: './section-overview.png'
code: 'nn-2'
order: 2
series: "llm-basics"
level: beginner
topic: "Neural Networks"
tags: ['neural-networks', 'training', 'backpropagation']
---

## Overview

![Concept overview: How Models Learn: Gradient Descent and Backprop in Plain Words](./section-overview.png)

GPT-3 has 175 billion adjustable weights. Nobody set a single one of them by hand.

[Last article](/blog/what-is-a-neural-network/) established that a network's entire knowledge is its list of weight values. This 1 answers the obvious follow-up: how do those values get found? The answer is 2 ideas: one you can picture as walking downhill, one that is pure bookkeeping. Together they train everything from digit readers to ChatGPT.

## Deep dive

### First: give the network a score

Training starts by defining failure numerically. Show the network an example whose answer you know, compare its output to the truth, and compute a **loss**, 1 number measuring how wrong it was. 0 means perfect; big means bad.

Now imagine a strange landscape. Each possible setting of the weights is a location; the loss at that setting is the altitude. Somewhere in this landscape are low valleys, weight settings where the network is usually right. Training is a search for them.

The catch: for a real model the landscape has billions of dimensions and you can't see any of it. You only know the altitude *where you're standing*.

### Gradient descent: walking downhill blind

Here's what you *can* do while blind on a hillside: feel which way the ground slopes under your feet, and step downhill. Repeat.


That is the entire algorithm, called **gradient descent**. The "slope under your feet" is the *gradient*. For each of the billions of weights it answers one question: *if I nudged this weight slightly, would the loss go up or down, and how steeply?* Take a small step for every weight in its downhill direction, and the loss decreases. Do it millions of times, and a network that started as random noise becomes a digit reader, or a language model.

The step size (the *learning rate*) is a genuine tuning art: too small and training takes forever; too large and you overshoot valleys entirely. But the concept stays this simple.

### Backprop: the bill for every mistake

1 important question remains: how do you *compute* that slope for a weight buried deep in the middle of the network? When the final answer is wrong, which of the 175 billion knobs is to blame, and by how much?

The answer is **backpropagation**, popularized for multilayer neural networks in a [1986 paper by Rumelhart, Hinton, and Williams](https://doi.org/10.1038/323533a0). Strip the calculus away and it is an accounting procedure:


1. Start at the output, where the error is directly measurable
2. Split that error backward through the last layer: each contributing neuron receives blame in proportion to how strongly it pushed the wrong answer
3. Repeat, layer by layer, until every weight in the network holds its exact share of the bill

The mathematical engine is the chain rule from first-year calculus, applied systematically. The result is remarkable: **1 forward pass plus 1 backward pass prices every weight's blame simultaneously**, many parameter derivatives in a coordinated reverse pass, with cost determined by the operations and saved intermediates. Without this trick, you'd have to nudge weights 1 at a time to see what happens; at billions of weights, that's not a slow method, it's an impossible 1.

### The loop, assembled

Put the pieces together and training is a 4-beat loop:

> **guess** (forward pass) → **score** (loss) → **assign blame** (backward pass) → **nudge** (gradient step)

Run it on 1 batch of examples, then the next, millions of times. That loop is what a "training run" is, and why training costs what it costs: every beat touches every weight, and frontier models run the loop over trillions of words. When headlines say a model took months on thousands of GPUs, they're describing this loop, executed at industrial scale.

It's also why the field cares so much about training *efficiency*: shave 20% off the loop's cost and you've shaved 20% off one of the largest compute bills in industry. That thread (same loop, run cheaper) is exactly where this blog's [performance series](/blog/what-does-an-ml-performance-engineer-do/) picks up.

### Different losses encode different questions

![Deep dive: Different losses encode different questions](./deep-dive-component-01.png)

The earlier phrase “0 means perfect” is a useful first cartoon, but it is not universal. A negative log-likelihood can remain positive even for a good predictor, and continuous-density log losses can sometimes be negative. The relevant question is whether an objective rewards the behavior we want under a clearly stated model.

For a regression example, squared error penalizes the square of the difference between prediction and target. Predicting 3 when the target is 5 gives error 4; predicting 1 gives error 16. Larger mistakes receive disproportionately larger penalties. Under independent Gaussian observation noise with fixed variance, this loss corresponds to maximizing a conditional likelihood.

For classification, cross-entropy penalizes low probability assigned to the observed class, so giving the correct class probability 0.8 incurs about 0.2231 nats of loss while giving it probability 0.1 incurs about 2.3026, and both predictions could still choose the same winning label in a larger class set even though their probabilistic quality differs.

This connection is developed in [maximum likelihood estimation](/blog/maximum-likelihood-estimation/). Optimization searches for parameters; the statistical objective defines which fitted behavior we are searching for. Backpropagation works with the chosen differentiable objective and does not decide whether the objective matches the task.

### A complete gradient update by hand

Take the smallest useful regression model: a prediction $$\hat y=wx$$ with 1 weight w, input x, and no bias. Let the target be y, and choose half squared error

$$
\mathcal L(w)=\frac12(wx-y)^2.
$$

The half is a convenient scale convention. Differentiating gives

$$
\frac{d\mathcal L}{dw}=(wx-y)x.
$$

Use input 2, target 6, and initial weight 1. The prediction is 2, residual minus 4, loss 8, and gradient minus 8. With learning rate 0.1, gradient descent updates the weight to $$w_{\mathrm{new}}=1-0.1(-8)=1.8$$.

The new prediction is 3.6 and the new loss is 2.88. One step improved this example, but that does not guarantee every step in a general training run decreases the full-data objective. Minibatch gradients are estimates, learning rates can be too large, and complex landscapes can contain difficult regions.

Here the exact best weight is 3. Setting the derivative to 0 finds it immediately. Iterative optimization becomes necessary when the parameter space and objective make an exact closed-form solution impractical. The tiny example shows the sign and scale of an update without requiring a picture of billions of dimensions.

### Backpropagation is the chain rule, not blame allocation

![Deep dive: Backpropagation is the chain rule, not blame allocation](./deep-dive-component-02.png)

Now compose 2 scalar stages: $$h=w_1x$$ and $$\hat y=w_2h$$. With the same half squared error, the output derivative is prediction minus target. The chain rule sends that derivative through the operations that produced the output:

$$
\frac{\partial\mathcal L}{\partial w_2}=(\hat y-y)h,\qquad
\frac{\partial\mathcal L}{\partial w_1}=(\hat y-y)w_2x.
$$

For input 2, target 6, and both weights initially 1, the prediction is 2 and both weight gradients are minus 8. A simultaneous learning-rate-0.1 update sets both weights to 1.8. The new prediction is 6.48 and loss about 0.1152.

All gradients in that step are computed using the same forward-pass parameters. Updating 1 weight and then computing another gradient from the already modified model describes a different procedure. Framework optimizers typically apply updates after the backward pass has accumulated the requested gradients.

The “bill” metaphor should not imply causal or moral responsibility. A derivative is a local sensitivity: how an infinitesimal parameter change affects the chosen scalar objective at the current setting. Parameters interact, so gradients are not a unique decomposition of an error into independently attributable shares.

A nonlinear activation introduces its own derivative into the chain. If that derivative is small, upstream gradients can shrink. If intermediate factors are large, they can grow. This is the mathematical reason architecture, normalization, initialization, and residual connections influence trainability.


The scalar update also gives a stability condition. For the earlier model with fixed nonzero input $$x$$, target $$y$$, and half squared loss, let $$w_*=y/x$$ and learning rate $$\eta>0$$. The parameter error after a gradient step is

$$
w_{t+1}-w_*=(1-\eta x^2)(w_t-w_*).
$$

Thus repeated updates converge for this single quadratic when $$0<\eta<2/x^2$$. With $$x=2$$, the range is $$0<\eta<0.5$$. At $$\eta=0.1$$, error contracts by 0.6 each step; the initial error minus 2 becomes minus 1.2, matching the weight update from 1 to 1.8. At $$\eta=0.6$$, the factor is minus 1.4: errors alternate sign and grow in magnitude.

This turns “overshooting a valley” into a checked mechanism rather than assuming a downhill direction guarantees a better finite step. Neural-network objectives have multiple directions with different curvature, and minibatch estimates introduce noise, so this scalar bound is not a universal learning-rate recommendation. Input scaling changes curvature even in this tiny model; normalization and adaptive methods can change update conditioning. Monitor a fixed diagnostic batch to test optimization behavior, then verify held-out performance independently. Backprop computes a local derivative accurately for the specified graph; choosing a useful finite update remains the optimizer's responsibility.

### Minibatches trade exactness for useful computation

![Deep dive: Minibatches trade exactness for useful computation](./deep-dive-component-03.png)

A full gradient over a large dataset can be expensive, so minibatch training averages or sums gradients from a smaller group of examples and then updates the parameters, and repeating that with different batches provides a noisy approximation to optimization of the overall objective.

Batch size affects more than memory use. Larger batches can improve hardware utilization but provide fewer parameter updates for a fixed number of examples, they change gradient noise and can require different learning-rate schedules, and a bigger batch is not automatically better statistical learning even when it improves examples processed per second.

Gradient accumulation computes several smaller microbatches before 1 optimizer step. To match a larger effective batch, scale losses consistently and avoid accidental updates between microbatches. Dropout, batch-dependent normalization, numerical rounding, and optimizer schedules can still make 2 implementations differ.

For token training, the denominator matters too. Averaging each microbatch equally when the microbatches contain different numbers of valid tokens is not necessarily equivalent to averaging over all valid tokens. Mask padding correctly and state whether the reported loss is per example, per sequence, or per token.

### Why training needs extra memory

![Deep dive: Why training needs extra memory](./deep-dive-component-04.png)

The forward pass creates intermediate activations needed to evaluate derivatives later. Training additionally stores gradients and optimizer state. An adaptive optimizer can keep running averages related to gradients for every trainable parameter, and mixed-precision training may keep higher-precision copies.

Activation checkpointing saves fewer intermediates and recomputes selected forward operations during the backward pass. It trades extra arithmetic for lower memory use. The mathematical training objective can remain the same while the implementation's memory and timing change.

It is also inaccurate to say that forward plus backward always costs exactly twice inference. Backward work depends on operations, which inputs require gradients, and saved or recomputed intermediates. Dense-layer training often needs additional matrix multiplications beyond the forward multiplication. Measure the actual pipeline rather than deriving a bill from a fixed multiplier.

### Common misconceptions about optimization

**“Backpropagation and gradient descent are the same.”** Backpropagation computes derivatives efficiently. Gradient descent uses them to update parameters. Other optimizers can use the same computed gradients.

**“A nonzero gradient means the model is globally wrong.”** It means the chosen objective has local sensitivity at the current parameters. A model can have useful predictions and still admit improvements, and a 0 gradient does not prove a global optimum.

**“Lower training loss guarantees better deployment behavior.”** It demonstrates better fit to the optimized data/objective. Held-out evaluation and task-specific measurements determine whether that improvement transfers to the situations where the model will be used.

## Conclusion

- Training = minimizing a loss by walking downhill in weight-space: feel the slope, step, repeat. That's gradient descent.
- Backprop is the accounting trick that computes every weight's slope in 1 backward pass, an efficient application of the chain rule that helped make multilayer networks practical to train.
- Everything about a training run's cost follows from the loop: guess → score → blame → nudge, times every weight, times trillions of examples.


For a useful debugging exercise, freeze a tiny batch and run repeated optimization steps on it. The training loss should usually fall when the model has enough capacity and the computation is correct. Failure suggests checking labels, masking, gradient flow, and learning rate before changing the architecture. Success only shows that the system can fit those examples; it says little about unseen data. Restore a separate validation set before evaluating generalization. This small experiment separates an implementation problem from a data or modeling problem and makes the learning loop observable without requiring an expensive full training run.

### Sources

- Goodfellow, Bengio, Courville, [Deep Learning, machine-learning basics](https://www.deeplearningbook.org/contents/ml.html) and [deep feedforward networks](https://www.deeplearningbook.org/contents/mlp.html).

- Rumelhart, Hinton, Williams (1986). ["Learning representations by back-propagating errors"](https://doi.org/10.1038/323533a0), *Nature*
- Michael Nielsen, [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/chap1.html), ch. 1–2 (CC BY-NC 3.0; the gradient-descent figure is redrawn from it)
- Kaplan et al. (2020). ["Scaling Laws for Neural Language Models"](https://arxiv.org/abs/2001.08361) (why the loop gets run at ever-larger scale)

---

*Part of the [LLM Foundations & Mathematics](/series/llm-basics/) learning path. Browse its published articles by topic.*
