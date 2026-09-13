---
title: 'CNN: How Machines Learned to See'
description: "A convolutional network reads images the way you'd search a photo with a magnifying glass — one small pattern at a time. Here's the idea that owned computer vision for a decade."
pubDate: 'Sep 12 2026'
heroImage: './cover.png'
code: 'arch-1'
series: 'llm-basics'
topic: 'Neural Networks'
tags: ['cnn', 'neural-networks', 'computer-vision']
---

A tiny 1998 network with 60,000 weights read a large share of America's handwritten bank checks. Its core trick is three lines of intuition — and it's the same trick that, scaled up, won ImageNet in 2012 and kicked off the deep learning era.

[Previous articles](/blog/what-is-a-neural-network/) built the plain network: layers of weighted votes. This article covers the first great *architecture* — a network shaped to fit its data. That idea, "shape the network to fit the structure of the problem," is the single most reusable lesson in this series, and it returns when we reach Transformers.

## The problem with plain networks and images

Feed a 1000×1000 photo into the fully-connected networks we've seen so far and you hit a wall: every neuron in the first layer connects to every pixel — a million weights *per neuron*. Worse, that design has to re-learn everything at every location: a cat-ear detector learned in the top-left corner knows nothing about cat ears in the bottom-right.

Images have structure a plain network ignores: **nearby pixels are related, and patterns repeat across locations**. A convolutional neural network (CNN) bakes exactly those two facts into its wiring.

## The magnifying glass

Instead of looking at the whole image at once, a CNN slides a small window — say 3×3 pixels — across the image, checking for one specific pattern at every position:

![Convolution: a small filter slides across the image, producing a map of where its pattern appears — the same few weights reused at every location](./convolution.png)

That window is called a **filter**, and it is just a tiny set of weights — a 3×3 filter has nine. The output is a *map* of where in the image the pattern appears. One filter might light up on vertical edges, another on a patch of orange, another on a curve.

Two enormous wins fall out of this design:

- **Weight sharing.** Nine weights cover the entire image, because the same filter is reused at every position. The million-weight problem collapses to dozens.
- **Translation tolerance.** A cat ear activates the same filter wherever it appears. Learn once, detect anywhere.

Nobody designs the filters, by the way. They're weights — [gradient descent and backprop](/blog/how-models-learn/) set them, exactly as before. Early filters reliably converge to edge and color detectors on their own.

## Stacking: from edges to ears to cats

The real power is layering, and it's the "votes about votes" story again with a spatial twist. Layer one's filters find edges. Layer two's filters slide over *layer one's maps*, finding combinations of edges — corners, textures, circles. Layer three finds combinations of those: an eye, a wheel, a beak. In between, **pooling** layers shrink the maps, so each successive filter effectively sees a wider patch of the original image.

![LeNet-5's pipeline: alternating convolution and pooling layers distill the image into features a small classifier can vote on — redrawn from LeCun et al., 1998](./lenet.png)

The figure above is (a redrawn version of) LeNet-5, [Yann LeCun's 1998 digit reader](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) — the design that read bank checks in production when "neural network" was still a dirty word in grant applications. Fourteen years later, [AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) was recognizably the same recipe — convolution, pooling, stacking — with more layers, ReLU activations, GPUs to train on, and a million-image dataset. Same idea, more scale: 60 thousand weights to 60 million.

## A worked example: nine weights detect an edge

Let's actually run a filter, because the arithmetic makes the magic mundane. Take this 3×3 filter — nine weights arranged as a grid:

```
 +1   0  -1
 +1   0  -1
 +1   0  -1
```

To apply it at a position, lay it over a 3×3 patch of pixels, multiply each weight by the pixel under it, and sum. Now slide it over two different patches (pixel values: 0 = dark, 9 = bright):

```
patch A (uniform):     patch B (vertical edge):
 5 5 5                  9 9 0
 5 5 5                  9 9 0
 5 5 5                  9 9 0
```

- **Patch A**: (+1×5+0×5−1×5) × 3 rows = **0**. Nothing to see.
- **Patch B**: each row gives +1×9 + 0×9 − 1×0 = 9, total **27**. Strong response.

This filter fires precisely where brightness drops from left to right — it is a *vertical-edge detector*, built from nine numbers. Rotate the weights 90° and you detect horizontal edges. Nobody chose these values in a real CNN: [gradient descent](/blog/how-models-learn/) discovers edge detectors (and color-blob detectors, and texture detectors) in the first layer of essentially every vision network ever trained, because edges are the most reusable evidence about what's in an image. When AlexNet's authors visualized their trained first-layer filters, the grid looked like a catalog of oriented edges and color patches — learned, not designed.

## Going deeper: pooling, stride, and the growing field of view

Two supporting mechanics complete the picture. **Pooling** (typically "max pooling") slides a small window that keeps only the strongest response in each neighborhood — shrinking the map, discarding exact positions, keeping "this feature occurred around here." That builds in a useful indifference: a digit shifted two pixels still classifies the same.

The subtler consequence is the **receptive field**. After one 3×3 convolution, each value "sees" 3×3 original pixels. Stack another 3×3 conv on the pooled map and each new value indirectly sees a much larger patch of the original image. Depth therefore buys *scope*: layer 1 sees strokes, layer 5 sees letterforms, layer 10 sees whole objects. That's the mechanical reason the edges→textures→objects hierarchy emerges — each layer literally looks at a bigger piece of the world, expressed in the previous layer's vocabulary.

One number to anchor the efficiency claim: AlexNet's five convolutional layers, which do nearly all the visual understanding, hold only ~3.7M of its 60M weights — the old-style fully-connected layers bolted on the end hold the rest. The convolutional idea does the seeing at ~6% of the parameter budget.

## Common misconceptions

**"CNNs are obsolete now that Transformers exist."** Vision Transformers lead many benchmarks, but CNNs still run in enormous volume — phone cameras, medical imaging, industrial inspection, autonomous-vehicle stacks — because they're efficient at small scale and their built-in translation tolerance means they need far less training data. Architectures retire from the frontier long before they retire from production.

**"The filters are hand-designed."** Pre-2012 computer vision really did hand-design features (SIFT, HOG — an entire field's worth of PhD theses). The deep learning revolution was precisely that [backprop](/blog/how-models-learn/) *learns* the features end-to-end, and learned features beat two decades of engineered ones by 11 points in one contest.

**"Convolution is fundamentally different math from a normal network."** It's the same multiply-add-squash — with two constraints bolted on: each neuron connects only to a local patch, and all patches share one weight set. Constraints, not new machinery. That framing is worth keeping, because it recurs: most "new architectures" are the same core network with different constraints encoding different assumptions about data.

## The lesson that outlived the architecture

CNNs dominated vision for a decade, and they still run in your phone's camera. But the deeper lesson is the one to carry forward:

**A CNN is a plain neural network with knowledge about its data wired into its shape.** Images are local and repetitive, so the network looks locally and reuses weights. The data's structure became the network's structure — and that made it radically more efficient than a general network of the same power.

Hold that thought. In a few articles, we'll meet text — where the structure is *"any word can relate to any other word, near or far"* — and see why CNN-style locality fails there, why the recurrent networks of the next article struggled too, and why the architecture that finally fit language's structure ended up conquering everything, images included.

## Why CNNs and GPUs found each other

There's a hardware subplot here that foreshadows this blog's other series. Convolution looks like a bespoke operation, but implementations unroll it into **giant matrix multiplications** — thousands of independent patch-times-filter products with no ordering constraints between them. That's precisely the workload GPUs were built for (originally to shade millions of independent pixels for games).

The numbers behind the 2012 moment: AlexNet trained on **two consumer GTX 580 gaming cards** (~$500 each) for about six days. The authors estimated the same run on the CPUs of the day would have taken months — long enough that nobody would have bothered iterating. The experiment only became *runnable* because an architecture whose core op was embarrassingly parallel met a mass-market chip built for embarrassingly parallel math. Neither was designed for the other; the fit was luck, then strategy. NVIDIA noticed what its gaming chips were being used for, invested in CUDA and cuDNN for neural workloads, and a graphics company became the most valuable AI company on earth.

Keep this pattern; it's the thesis of the [Efficient AI series](/blog/blackwell-to-rubin-memory-math/): **architectures win when they fit the hardware of their moment, and hardware evolves toward the architectures that win.** CNNs-meet-GPUs was the first round. Transformers-meet-tensor-cores was the second. Whatever wins next will fit the silicon of 2030.

## Takeaway

- Plain networks waste millions of weights on images and re-learn every pattern per location; CNNs fix both with one move — a small filter slid across the whole image.
- Weight sharing (nine weights, million positions) plus stacking (edges → textures → objects) is the entire recipe, from 1998's LeNet to 2012's AlexNet.
- The transferable lesson: match the network's shape to the data's structure. That principle picks the winners in every architecture era, including the Transformer's.

## Sources

- LeCun, Bottou, Bengio, Haffner (1998). ["Gradient-Based Learning Applied to Document Recognition"](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) (LeNet-5; the pipeline figure is redrawn from it)
- Krizhevsky, Sutskever, Hinton (2012). ["ImageNet Classification with Deep Convolutional Neural Networks"](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (AlexNet)
- Stanford CS231n — [Convolutional Networks notes](https://cs231n.github.io/convolutional-networks/) (the standard visual treatment)

---

*Part of the **Fundamental of LLM** series. Previous: [How models learn](/blog/how-models-learn/). Next: RNNs and LSTMs — how machines learned sequences, and why they hit a wall.*
