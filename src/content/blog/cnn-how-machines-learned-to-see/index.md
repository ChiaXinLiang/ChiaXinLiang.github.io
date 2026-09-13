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

## The lesson that outlived the architecture

CNNs dominated vision for a decade, and they still run in your phone's camera. But the deeper lesson is the one to carry forward:

**A CNN is a plain neural network with knowledge about its data wired into its shape.** Images are local and repetitive, so the network looks locally and reuses weights. The data's structure became the network's structure — and that made it radically more efficient than a general network of the same power.

Hold that thought. In a few articles, we'll meet text — where the structure is *"any word can relate to any other word, near or far"* — and see why CNN-style locality fails there, why the recurrent networks of the next article struggled too, and why the architecture that finally fit language's structure ended up conquering everything, images included.

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
