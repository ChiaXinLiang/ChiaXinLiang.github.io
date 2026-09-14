---
title: "DPU Architecture: Network, Storage, and Infrastructure Isolation"
description: "Trace a packet or storage request through queues, parsing, DMA, offload engines and embedded CPU control."
pubDate: "2026-09-13"
heroImage: "./section-overview.png"
series: "comp-arch"
code: "dpu-1"
order: 21
topic: "Modern Accelerators"
tags: ["Computer Architecture", "dpu"]
---

## Overview

![Concept overview: Trace a packet or storage request through queues, parsing, DMA, offload engines and embedded CPU control](./section-overview.png)

A data processing unit, or DPU, is an infrastructure processor: it combines network-facing data paths, programmable control and offload engines, and its job is the data moving between machines, which means transforming packet and storage bytes, managing queues, and copying authorized bytes from one memory to another. None of that is a matrix multiply. NVIDIA's BlueField-4 is one product in this category. A DPU can speed up an AI system without ever touching the neural network.

In the overview, solid arrows carry data and dashed ones carry management decisions. The host submits work and reads results back, the DPU's control software sets the rules, and the NIC and offload logic carry them out, though which features exist and which software owns them depends entirely on the product and the operating mode. "DPU" names a category, not an instruction set.

This article follows one packet and one storage transfer through the chip to show what work happens, where it happens, and why that split matters to an AI cluster. The examples are illustrative budgets, not measurements. The snapshot is dated 2026-09-13, and the principles carry over to SmartNIC designs whose documented capabilities match.

## Deep dive

### Follow a packet through the data path

![Deep dive: Follow a packet through the data path](./deep-dive-component-01.png)

The packet figure starts the moment a bit arrives on the wire, then walks through parsing, classification, queue selection and DMA. The parser picks out fields like protocol headers, a match/action stage applies the rules you configured, queue logic chooses the destination stream and manages descriptors, and DMA copies the payload into an authorized region before a completion record tells software what arrived.

Each stage does a different kind of work: the parser reads fields at fixed offsets and checks the lengths make sense, then the classifier looks those fields up in a rule table, address generation adds an offset to the buffer address, and the scheduler picks which request goes next. Notice what is missing. Not one of these steps is a matrix multiply: the packet belongs to an AI job, but receiving it is lookups and arithmetic on a few bytes of Ethernet header.

A packet arriving does not mean the application can read a complete tensor. Software has to see the right completion first, plus whatever memory-ordering guarantees the platform requires, and truncated packets, malformed headers, unsupported encapsulation and exhausted queues all need documented error behavior. If the parser trusts a length field before checking its bounds, every address it computes downstream can be wrong.

Work a packet-rate budget. Take a 100-Gb/s stream and 1,500-byte payloads: dividing $$100\times10^9$$ bits/s by $$1,500\times8$$ bits gives approximately 8.33 million payloads/s, which leaves 120 ns of average payload serialization time, and Ethernet headers, preamble and inter-packet gaps all cut into the rate you can actually reach.

120 ns is not much time. That is why a CPU-only per-packet path gets expensive: software must inspect, schedule and move each item inside that window, so hardware pipelines handle the regular stages instead while the programmable cores take the exceptions. [NVIDIA's DPU platform description](https://www.nvidia.com/en-us/networking/products/data-processing-unit/) frames network, storage and security offload as infrastructure responsibilities; the parser stages and budget above are an independent explanatory model.

### What the offload engines calculate

![Deep dive: What the offload engines calculate](./deep-dive-component-02.png)

The engine figure lists the transformations separately because their algorithms and costs differ. A checksum walks the bytes and accumulates a value under a protocol rule, the crypto engine implements a specific cipher or authentication primitive, and a compressor encodes repeated structure in a supported format. Address translation and access checks decide where a transfer may land.

Some stages pipeline happily over streaming bytes. Others need message boundaries, state or metadata first: an encryption engine has to honor the algorithm's nonce, key and authentication rules, and a compressor is harder to budget still, because its output length depends on both the input and the format, so you cannot assume a payload shrinks by a constant factor. The 0.6 output bytes per input byte used below came from one dataset, not from the algorithm. The controller has to handle incompressible data and reserve enough room at the destination.

Offload moves an operation to another engine. It does not remove the dependencies: someone still provisions the keys, manages the rules, and gets the applications to agree on formats, and an unsupported packet falls through to an exception path where embedded software or the host picks it up. Watch that exception rate, because it shapes the whole system even when the fast path stays efficient.

Suppose a storage service takes in 10 GB/s of uncompressed data, and the chosen method produces 0.6 output bytes per input byte on one measured dataset. The stream leaving it would be 6 GB/s before framing, which needs two things at once: a compressor that keeps up with the input rate, and a destination that keeps up with the output rate. Neither follows from the compression ratio.

NVIDIA's [BlueField-4 announcement](https://blogs.nvidia.com/blog/bluefield-4-ai-factory/) describes a product combining networking with programmable infrastructure processing. Read it as a dated example, not as evidence that every DPU carries that exact CPU, link rate or engine set. Start from the supported transformations and their data contracts. Then ask what decides it: does offloading them cut host work, or improve isolation, for the application you have?

### Storage and RDMA data movement

![Deep dive: Storage and RDMA data movement](./deep-dive-component-03.png)

The storage figure keeps the command separate from its payload. Software submits a request naming the data and a destination, transport logic talks to the far endpoint, a memory operation moves the bytes, and a completion reports the result. Remote direct memory access, or RDMA, can skip repeated CPU copying on a supported path, but it still needs queues, descriptors, permissions and synchronization.

A remote memory operation needs an authorized region and the right credentials. "Direct" does not mean unrestricted. The receiving side has to know which buffers exist, how long each stays valid, and when it can be reused, because deregistering a buffer while a request is still outstanding gives you a lifetime bug.

Take a tensor transfer. 16 MiB is 16,777,216 bytes, and at a sustained 25 GB/s payload rate, serialization alone costs about 0.671 ms, plus request startup, queueing, transport processing and completion visibility. A link's nominal bit rate is not this sustained payload rate. Split the operation across several requests and it pays the per-request overhead again each time.

The storage path may add integrity checks, encryption, compression or namespace translation, and whether any of those run on the DPU depends on the stack. Storage semantics still govern durability, ordering and failure. A local DMA that succeeded is not proof that a remote durable write completed.

An AI system gains here when input loading, checkpoints or distributed memory movement eat a large share of host resources, though how much depends on placement and software integration: a compute-bound matrix kernel will not speed up just because storage handling moved off the host. Measure host CPU load, delivered bytes, queue latency and end-to-end completion separately. Report the DPU's contribution in the metric it actually changed, not as a throughput claim the data does not support.

### Isolation and control ownership

![Deep dive: Isolation and control ownership](./deep-dive-component-04.png)

The isolation figure draws a trust boundary around infrastructure ownership and marks the permission checks. A DPU with its own managed control environment can keep network and storage administration away from whatever the host runs, which helps when the host runs tenant workloads and the infrastructure services need a different owner.

The boundary is a system property. Adding a chip does not create it. Firmware, boot configuration, management access, IOMMU mappings, memory registration, keys and permissions all feed into it, and any configuration the host can change itself weakens the separation. A privileged management path needs its own authentication and update policy.

Keep the control plane separate from the data plane. The control plane decides rules, allocates resources and provisions credentials, while the data plane processes packets and transfers under those rules. The dashed arrows carry configuration, not payload: the embedded CPU does not copy every byte.

An IOMMU translates and constrains device memory access according to its configured mappings, and RDMA memory keys add another form of authorization inside the transport that uses them. Neither one alone gives you tenant isolation: their scopes differ, and the platform still has to stop an untrusted owner from editing the configuration.

A useful review traces 4 questions. Who can install a rule? Who can reach the buffer? Who observes completion, and who can reset the device? Then test the failures: invalid descriptors, disallowed regions, queue exhaustion, recovery after reset. These checks let you explain infrastructure processing without assuming proprietary enforcement details, and they connect back to the FPGA project's [command validation](/blog/fpga-ai-control-1-command-registers-and-scheduling/) and DMA models, where address bounds and completion are explicit parts of correctness.

### Build a complete service budget

A service can have a fast packet path and still be limited by memory writes, descriptor updates or its exception handler, so budget the whole path in the same units. Say traffic arrives at 20 GB/s but the destination sustains only 12 GB/s of writes: a queue absorbs the difference, but only for a while, and a 1-GB free buffer fills in $$1/(20-12)=0.125$$ seconds under this simplified constant-rate model. Adding buffer changes when backpressure starts. It does not fix a steady-state mismatch.

Record payload bytes separately from wire bytes and descriptor traffic, and remember that a multiport device's aggregate rate tells you nothing about one flow's usable rate. Some workloads send many small messages, where per-request processing dominates. Others send large contiguous transfers. Keep that distribution in the benchmark instead of collapsing it to one average size such as the 1,500-byte payload above, and the result tells you where the bottleneck really is: packet rate, payload bandwidth, queue management or downstream storage.

### A worked engineering decision

Follow a storage request from a remote client to a host application. Its path may include packet reception, transport handling, access control, encryption, DMA and a storage protocol. Moving one stage onto a DPU changes where the work happens without removing a copy, a context switch or a synchronization point. Draw the real path, marking where ownership changes hands and where the host may read the data.

Then classify each stage. Header matching may use a lookup or a match/action pipeline, encryption may use a dedicated engine, and policy logic may run on the embedded cores, while tensor computation stays a separate question entirely. Lumping all of it together as "AI compute" hides which resource is actually busy, and a request rate and a cryptographic byte rate are different metrics resting on different packet-size and batching assumptions.

For the experiment, hold the security policy and observable application behavior identical on both paths. Measure end-to-end latency and host resources, not just engine activity on the accelerator, and count the extra data movement, queue depth and completion processing. Setup cost dominates small requests. Large streams, like the 16 MiB transfer above, stress sustained byte processing instead. Failure and retry behavior must preserve the same correctness and isolation contract.

Finally, ask whether removing infrastructure work from the host frees a resource the application can use. If the host already waits on another bottleneck, freeing CPU cycles may raise capacity or improve isolation without moving inference latency at all, so report that precisely. The DPU's value is where it puts the work: infrastructure processing sits at the data boundary, with capabilities and isolation rules set by the system around it.

## Conclusion

A DPU computes on infrastructure data: headers, descriptors, bytes, addresses and queue state. Its architecture pairs regular offload pipelines with programmable control and memory movement, a role that complements the GPU and TPU instead of duplicating their matrix datapaths. The 120-ns average payload window from the packet budget above is why the regular stages belong in hardware and the exceptions belong in software.

To judge a DPU, follow one representative request from submission, through the authorized transfer, to completion, identifying which operations really get offloaded and which software owns their configuration. Then measure delivered bytes, host work and completion latency under the failure behavior you intend to run. No processor label or nominal network rate substitutes for that trace.

### Sources

- [NVIDIA data processing unit platform](https://www.nvidia.com/en-us/networking/products/data-processing-unit/)
- [BlueField-4 announcement](https://blogs.nvidia.com/blog/bluefield-4-ai-factory/)
