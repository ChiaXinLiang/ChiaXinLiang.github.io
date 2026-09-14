---
title: "Network Failures and Stragglers: Diagnosing Distributed Job Stalls"
description: "Distinguish slow participation from failed transport, derive group-tail amplification, and connect bounded failure detection to safe checkpoint-based recovery."
pubDate: "Sep 13 2026"
updatedDate: "Sep 13 2026"
heroImage: './section-overview.png'
series: "ai-networking"
code: "network-8"
order: 11
topic: "Congestion, Expert Dispatch, and Reliability"
level: "advanced"
tags: ["ai-networking", "ai-infrastructure"]
---

## Overview

![Concept overview: Network Failures and Stragglers: Diagnosing Distributed Job Stalls. Several ranks approach a collective synchronization point.](./section-overview.png)

A distributed job can stall while every process remains alive and every GPU still appears allocated. One rank may be late, another may have failed asynchronously, or a collective may be waiting for a participant that entered a different operation. A network-path failure is one possible cause among several.

The challenge is to find the blocked dependency before a cascade of timeouts buries the original event. Liveness checks show that a process responds. Progress checks show whether the group advances through the expected computation. Correlated rank and path evidence connects the two.

This article builds a diagnosis and recovery method for distributed stalls. The probability calculations are illustrative models, not measured cluster failure rates. Timeout, tracing, and restart support depends on the installed framework and communication stack.

## Deep dive

### 1. Define useful progress at a group-wide boundary

![Deep-dive illustration: Define useful progress at a group-wide boundary](./deep-dive.png)

Choose progress events with clear semantics: completed optimizer updates, completed logical phases, or durable checkpoints. A rank that prints logs or answers a health query has not proved that the whole group completed another update.

Record the last known common step and the current phase on each rank. Include logical collective sequence numbers where controlled instrumentation supports them. The goal is to find where the ranks diverged, not just which process eventually reported a timeout.

A simple group-progress indicator can be the minimum completed logical step across ranks:

$$
s_{\mathrm{group}}=\min_r s_r.
$$

This bookkeeping assumes the step counters describe the same protocol boundary. Counters that increment independently and mean different things cannot be combined safely. The minimum can help find a lagging participant, but it does not certify a recoverable checkpoint.

Separate "no progress" from expected long work. A very large prefill, checkpoint save, compilation, or input operation may legitimately take longer than ordinary iterations. The detector needs phase context; it should not treat every interval without a step update as a network failure.

### 2. Group tails amplify individual variability

![Deep dive: 2. Group tails amplify individual variability](./deep-dive-component-01.png)

If a synchronized phase waits for every participant, the slowest relevant rank shapes its completion time. So even modest per-rank variability can create a large group-tail effect as the group grows.

Under an illustrative independence model with identical per-rank completion distribution F(t), the maximum's distribution is

$$
P(T_{\max}\le t)=F(t)^p.
$$

If each rank has a 1% chance of exceeding a threshold, the probability that at least one of 128 independent ranks exceeds it is 1 minus 0.99 raised to 128, about 72.4%. This is not a prediction for a real job; rank delays can be correlated and distributions can differ.

The model explains why single-rank p99 is not enough to describe a large synchronized group's tail. Measure the actual per-rank and group populations. Shared storage, fabric congestion, and host scheduling can create correlated delays that need different analysis than independent noise.

Check which rank is late across repeated events. A consistently late placement suggests a persistent path or resource issue. A changing late rank can suggest workload variability, shared contention, or a broader scheduling problem. These patterns guide the next test, but they do not prove the cause by themselves.

### 3. Separate late readiness from stalled communication

Trace the computation before the collective and the entry into it. A rank waiting on input or a long kernel has not yet supplied its contribution. Peers can spend a long time inside communication even though the network is not the original bottleneck.

Compute readiness skew from comparable per-rank events: the latest readiness time minus the earliest. Also check progress after the required participants are ready. This separates participation delay from transfer or protocol delay, within the limits of the tracing method.

Take an illustrative event: 7 ranks become ready at 20 milliseconds while one becomes ready at 80 milliseconds. A collective finishing at 90 milliseconds gives early ranks a 70-millisecond interval, but much of that time passes before the final participant is ready.

If the late rank never enters the expected operation, check collective ordering, conditional branches, and earlier failures. A mismatch in group, count, dtype, or operation can create a permanent wait. Raising the timeout cannot make incompatible distributed participation correct.

### 4. Preserve the first causal error

Asynchronous device errors and transport failures can surface after the operation that caused them. A launcher may then kill peers, producing many secondary errors. Keep timestamps and rank context around the earliest relevant event; do not focus only on the final shutdown message.

Collect bounded logs from all ranks in the failing group. Include phase and operation sequence so you can compare events logically even when cross-host clocks are imperfect. Monotonic local durations stay useful; exact global ordering needs a documented clock relationship.

Check device error reports, process exits, communication diagnostics, and adapter or link events. A rank that stopped after a kernel failure can leave others waiting in a collective. A path-down event that lines up with communication loss supports a different hypothesis.

Do not assume the first printed message is the cause. Logging can be buffered, and errors can propagate differently across ranks. Use the protocol sequence and correlated evidence to find the earliest failed dependency the observations actually support.

### 5. Use path evidence to test a network hypothesis

Map the affected ranks to GPUs, adapters, hosts, and relevant switch boundaries. Compare with healthy ranks and paths. A stall limited to one adapter or cut can point to a local resource or connectivity problem. Broad, simultaneous degradation can point to a shared dependency.

Check link state, error and retry counters, congestion evidence, and transport selection with supported diagnostics. Counter increases need a defined interval and traffic population. Historical totals can include unrelated events, so do not blame them on the current stall by default.

After saving the failure evidence, reproduce with controlled host-buffer and GPU-buffer tests. If both fail across the same boundary, the external path becomes a stronger candidate. If only GPU buffers fail, look at registration, locality, and device synchronization.

A healthy isolated test does not rule out an intermittent failure or congestion under concurrency. Match message sizes, rank groups, and traffic timing where possible. A reproduction that removes the original burst or placement can remove the symptom along with the suspected cause.

### 6. Distinguish health checks from progress checks

A heartbeat can show that a process or endpoint responds. It may keep going during a blocked collective if another thread handles the heartbeat. Device queries can also respond while the application is stalled. These signals help, but their scope is limited.

A progress detector should watch protocol events that match the job phase. It can track how long the group sits at one logical boundary and compare that duration with an expected range. Include context for checkpointing, compilation, and unusually large work.

A simple detection delay budget is

$$
T_{\mathrm{detect}}\approx T_{\mathrm{observation\ interval}}+T_{\mathrm{threshold}}+T_{\mathrm{reporting}}.
$$

This approximation shows the tradeoff between fast detection and false alarms. It does not give a universal threshold. Real detectors may use multiple missed observations, adaptive baselines, or framework-specific failure signals.

Test false-positive behavior under legitimate long phases. A detector that keeps killing healthy jobs can hurt goodput more than the occasional stall it catches. Report both detection latency and the wrong-kill rate under representative workloads.

### 7. Timeouts bound waiting but do not repair the cause

A timeout defines when the current policy gives up on an operation. A longer timeout tolerates expected variability, but it also keeps wasting allocated resources after a permanent failure. A shorter one detects failures sooner but can reject healthy slow operations.

Set policy from observed phase distributions and the system's supported failure behavior. Keep operation timeout, launcher supervision, and service deadline separate. They may start at different boundaries and trigger different cleanup paths.

Do not let surviving ranks continue with their local tensors after a partially failed collective unless the framework explicitly supports that protocol. Their state can diverge, and communicator membership or outstanding operations may no longer be valid.

Record the reason for termination, and make sure cleanup releases devices, adapters, buffers, and scheduler allocations through supported paths. A failed job that leaves stale resources can break later jobs and make recovery look unreliable for a different reason.

### 8. Recover from state with defined consistency

![Deep dive: 8. Recover from state with defined consistency](./deep-dive-component-03.png)

For training, recovery normally restores a supported checkpoint with the state needed to continue the optimization method. Model weights alone may miss optimizer, scheduler, random-number, and data-progress state. Document the exact reproducibility target.

A durable checkpoint is not the same as a queued asynchronous save. Recovery should use a completed state whose metadata and shards describe a consistent logical point. A directory's existence does not prove that all required data reached durable storage.

A simplified interruption cost is

$$
T_{\mathrm{lost}}=T_{\mathrm{detect}}+T_{\mathrm{cleanup}}+T_{\mathrm{restart}}+T_{\mathrm{restore}}+T_{\mathrm{replayed}}.
$$

Some components can overlap, so this serial model is a starting point for accounting. Measure useful recovered progress as well as startup completion. A healthy process does not prove that the restored job ran its next optimizer update.

Use the [checkpoint and recovery method](/blog/distributed-checkpoints-recovery-goodput/) to define state and test restart. Repeat recovery tests with the supported world-size or sharding changes if those are part of the operating policy.

### 9. Test failure behavior with controlled scope

![Deep dive: 9. Test failure behavior with controlled scope](./deep-dive-component-02.png)

Use a dedicated test environment and supported fault mechanisms to exercise worker termination, transport interruption, delayed participation, and unavailable checkpoint storage where relevant. Keep the experiment bounded and away from unrelated production traffic.

For each case, verify detection, first-error preservation, cleanup, restart, restored consistency, and useful progress. A test that ends when the launcher exits verifies only termination, not recovery. A test that ends when workers start verifies only part of restart.

Include slow-but-healthy cases to test false positives. Delay input or add representative long computation without changing collective ordering. Compare how the detector handles these cases against true failed participation and path loss.

A deterministic delayed-rank test can hold one rank before a known collective, record the others' wait, then release it. The expected result should still be correct after completion. A mismatched-operation test has different semantics: it should fail under the supported detector, not get released as ordinary latency. Keeping these cases distinct verifies that diagnostics classify the blocked dependency instead of calling every wait a broken network.

### 10. Convert incidents into regression cases

Keep a compact incident record: the workload, last common progress, rank divergence, first supported causal evidence, failing layer, recovery action, and verification result. Keep the minimal reproducer and the relevant topology mapping too.

Measure goodput over an interval that includes detection and recovery. High utilization before the failure and a successful restart afterward can still come with a lot of lost useful work. The end-to-end metric should count durable or otherwise accepted progress, judged by the job's objective.

Revisit detection policy after changes to model size, checkpointing, hardware, or communication scheduling. Healthy phase durations can change, and a static threshold can become either too permissive or too aggressive.

## Conclusion

Distributed stalls are dependency failures that need group-wide evidence. Find the last common progress, separate readiness from transport, preserve the first causal error, and recover through a supported consistency boundary. The network is an important layer of that investigation, but reliable diagnosis begins with what every participant was required to do next.

### Sources

- [NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html).
- [PyTorch distributed communication and debugging](https://docs.pytorch.org/docs/stable/distributed.html).
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html).
