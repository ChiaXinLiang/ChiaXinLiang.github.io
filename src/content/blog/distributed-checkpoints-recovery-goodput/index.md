---
title: "Distributed Checkpoints and Recovery: Goodput Under Failure"
description: "Define recoverable training state, derive checkpoint intervals, and verify distributed save/load without confusing asynchronous completion with durability."
pubDate: "Sep 13 2026"
updatedDate: "Sep 13 2026"
heroImage: "./cover.png"
series: "distributed-training"
code: "train-8"
order: 8
topic: "Distributed Training"
level: "advanced"
tags: ["distributed-training", "ai-infrastructure"]
---

A training cluster can display high GPU utilization while making little durable progress. If checkpoints take too long, they repeatedly interrupt computation. If checkpoints are too infrequent, failures discard many completed steps. If recovery is not tested, a directory full of files can turn out not to contain a usable continuation of the training job.

Distributed checkpointing is therefore part of performance engineering, not only a reliability feature. It defines what work survives, how much service time saving consumes, and how quickly the job returns to useful training after an interruption. The appropriate metric includes checkpoint and recovery costs alongside ordinary step throughput.

We will define the state needed for a meaningful restart, derive a simplified checkpoint interval, and examine asynchronous saving and resharded loading. The reliability calculations assume a stationary failure process and are illustrative. Actual intervals should use observed job-level behavior and storage performance under the intended workload.

## 1. Define the training point being preserved

![Concept overview: Distributed Checkpoints and Recovery: Goodput Under Failure. A distributed GPU job writes coordinated checkpoint shards into storage.](./section-overview.png)

*Overview of the article’s core mechanism. The following sections explain the objects, relationships, equations, assumptions, and worked examples shown here.*


A model-weight checkpoint is useful for inference or evaluation, but continuing training usually requires additional state. The optimizer needs its moments and other persistent values. A learning-rate scheduler needs its position. Mixed-precision machinery may maintain scaling state. The loop needs its optimizer-step count and accumulation position.

Random-number state can matter for dropout and sampling. The input pipeline needs enough information to reconstruct data progress, depending on the dataset, shuffling strategy, and reproducibility target. Some systems can restart with a valid but not exactly identical future sample stream; that should be documented rather than silently described as bitwise continuation.

The checkpoint also needs configuration and version information sufficient to interpret tensor shapes, numerical representations, and distributed ownership. A shard without its layout metadata may be insufficient to reconstruct the logical model. A collection of independently captured ranks can describe different training points if saving is not coordinated correctly.

Choose a clear capture boundary, such as immediately after a completed optimizer update. Capturing in the middle of accumulation requires preserving the partial gradients and loop state consistently. The simplest supported boundary is often preferable to a more flexible boundary whose recovery semantics are not understood.

## 2. Sharded state should not require an impossible gather

A fully sharded training job can hold state that no single GPU can materialize completely. Gathering all parameters and optimizer tensors onto rank 0 just to save them can therefore recreate the original capacity problem. Distributed writers instead save local shards alongside metadata describing their logical ownership.

PyTorch Distributed Checkpoint supports parallel save and load and documents load-time resharding. That capability depends on supported representations and planners; it does not imply arbitrary compatibility across every optimizer, framework, or historical checkpoint format.

The storage path includes metadata operations, local serialization or staging, shard writes, and final completion signaling. If many ranks write simultaneously, aggregate throughput can be limited by shared storage targets or network paths. Individual-device write rates do not add without bound.

Count both logical checkpoint bytes and actual physical bytes written. Replicated tensors, duplicated metadata, format overhead, and compression can change the relationship. A filesystem reporting a large directory size and a network reporting transferred bytes may use different conventions, so compare measurements under defined units.

![Deep dive: 2. Sharded state should not require an impossible gather](./deep-dive-component-02.png)


## 3. Derive a simple checkpoint interval

Let I be useful compute time between checkpoints, C the blocking checkpoint cost, tau the mean time between job-level interruptions, and R the average recovery cost. Under a simplified independent, stationary failure model, expected waste can be approximated by

$$
W(I)\approx\frac{C}{I}+\frac{I}{2\tau}+\frac{R}{\tau}.
$$

The first term is checkpoint overhead. The second assumes a failure loses about half an interval on average. The third accounts for time spent recovering. The approximation assumes the costs are relatively small and does not describe every failure distribution or asynchronous schedule.

Differentiating the interval-dependent terms gives

$$
I_{\mathrm{opt}}\approx\sqrt{2C\tau}.
$$

The result is a useful planning baseline: faster checkpointing permits more frequent saves, while more reliable jobs justify longer intervals. It is not a universal optimal policy when failure risk changes over time, planned preemptions are announced, checkpoint bandwidth varies, or the job is nearly finished.

Use the job-level interruption interval, not an unrelated component reliability number. If one device failure stops a large gang-scheduled job, the effective job failure process can differ greatly from the reliability of one GPU. Shared power, networking, and software failures also violate naive independence assumptions.



![Deep-dive illustration: Derive a simple checkpoint interval](./deep-dive.png)

## 4. Work a checkpoint-cost example

Suppose a blocking checkpoint costs 30 seconds, average job interruption time is 7200 seconds, and recovery costs 45 seconds. The simplified optimal interval is the square root of 432,000, approximately 657 seconds, or about 11 minutes of useful computation.

At that interval, checkpoint overhead is about 30 divided by 657, or 4.6%. Expected lost work is about 657 divided by 14,400, also 4.6%. Recovery contributes 45 divided by 7200, or 0.625%. Total approximate waste is therefore about 9.8%.

These values are hypothetical and demonstrate the model’s sensitivity. Reducing checkpoint cost to 5 seconds changes the interval estimate to about 268 seconds. Although saves become more frequent, the combined checkpoint and lost-work terms shrink. The speed of the storage path consequently affects durable training goodput.

A long interval chosen solely to minimize visible save stalls can produce worse expected progress when failures are common. A very short interval chosen solely to minimize lost steps can consume excessive storage service. The interval should be justified with both measured checkpoint behavior and a documented interruption model.

## 5. Asynchronous saving separates capture from persistence

An asynchronous checkpoint can copy or stage a consistent state and write it while training resumes. This reduces the visibly blocking interval, but the background work still consumes host memory, storage bandwidth, and potentially network or device-transfer resources.

A returned handle or queued request is not necessarily a durable checkpoint. Define separate timestamps for capture completion, background write completion, and committed recoverability according to the storage and framework guarantees. The job must know which checkpoint is safe to use after an interruption.

If training mutates tensors while the writer still reads them, saving can capture inconsistent values unless staging or another ownership mechanism prevents the race. A background writer holding references to live mutable tensors is not automatically a correct asynchronous checkpoint implementation.

Limit the number of outstanding saves. If checkpoint production is faster than the writer can persist them, queues and staging buffers grow. That can consume host capacity and turn an apparently asynchronous feature into an eventual stall. Backpressure is part of the scheduling design, not evidence that asynchronous saving failed conceptually.

## 6. Measure background contention, not only blocking time

A checkpoint writer can compete with the training input pipeline, optimizer offload, and other jobs for storage and host resources. Training step time may increase during background writes even when no explicit save barrier appears in the GPU trace.

Measure ordinary steps and checkpoint-overlapping steps separately, then evaluate progress across the full interval. If compute-only step time is T_0 and background periods produce slower T_1, the difference belongs in the checkpoint cost model. Counting only the short staging pause understates the performance impact.

The relevant checkpoint cost for interval planning is therefore an effective lost-progress contribution, not necessarily the writer’s wall-clock duration. A writer can take 60 seconds in the background while costing much less than 60 seconds of useful training, or it can heavily interfere with another critical service.

Capture host-memory peaks and queue depth alongside bandwidth. Staging large optimizer state can exceed available DRAM even when GPU memory is comfortable. The worst-case supported checkpoint path should be part of capacity testing for a long-running job.

## 7. Resharded loading is an algorithmic operation

Loading a checkpoint into a different rank layout requires mapping logical tensors to new owners. Metadata describes the old shards; the load planner determines which ranges each new rank needs. Supported systems can avoid reconstructing every complete tensor on one device.

A changed layout can also change process-group definitions, optimizer partitioning, and local buffer sizes. Successful tensor loading does not guarantee the subsequent distributed computation is valid. The new training configuration must satisfy shape, divisibility, and collective-ordering requirements.

Test the exact layout changes that operations will support, such as restarting with a different data-sharding degree. Do not generalize one successful resharding test to arbitrary tensor or pipeline partitions, optimizer formats, or software migrations. Compatibility is a specific capability with documented limits.

A useful recovery test loads the checkpoint, checks the logical training position and tensor values, executes forward and backward, and completes an optimizer update. This verifies more than file readability. Include the longest supported input shape when that path can recreate memory peaks absent from a small test.

## 8. Make incomplete checkpoints distinguishable

A checkpoint should have a clear completion or commit convention. Readers must be able to distinguish a fully written state from a partially created directory left by interruption. Atomicity depends on the storage and writer design; a filename or timestamp alone is not a proof of consistency.

Keep metadata and shard references coherent. Validate that the expected objects exist and can be interpreted before advertising a checkpoint as recoverable. Checksums or format-specific validation can help detect corruption, but they do not prove that all ranks captured the same logical training point.

Retention policies should preserve enough known-good checkpoints to recover from an incomplete or unusable newest save. The appropriate number depends on storage capacity, write cost, and operational requirements. Test recovery from an older completed checkpoint as well as from the latest one.

Treat cleanup and lifecycle separately from training performance. Removing obsolete state should not race with a writer or reader still using it. Operational policies need explicit ownership and completion signals, just as device buffers do in an asynchronous compute schedule.

## 9. Evaluate durable training goodput

Raw tokens per second counts useful computation while the job is active. Durable progress over an observation interval also accounts for checkpoint stalls, background slowdown, lost work, and recovery. A simple empirical metric divides retained useful training tokens by elapsed wall-clock time.

Choose the retention definition carefully. Tokens computed and later replayed after rollback should not be counted twice as durable progress. If a recovery intentionally changes the data stream, explain how the training position and useful-work accounting are reconstructed.

Run failure-injection tests in an appropriate disposable training environment before relying on the recovery path. Exercise interruption during computation and during checkpoint writing, then verify the selected completed checkpoint and continued optimizer updates. A successful steady-state benchmark cannot establish failure behavior it never exercised.

Keep recovery results with the training configuration. Record checkpoint format, storage backend, software versions, rank layout, capture boundary, and supported restart scenarios. The method should remain reviewable when a later change alters optimizer state or distributed ownership.

## Takeaway

A recoverable checkpoint represents one consistent training point, including the state required to continue the intended update sequence. Distributed writers preserve sharded ownership; asynchronous saving separates capture from durable completion but introduces background resource costs.

Choose intervals using measured effective save cost and job-level interruption behavior. Verify loading and a subsequent update, and report durable useful progress across failures. Checkpoint engineering improves the training result by preserving work, not simply by producing files faster.

## Sources

- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html): parallel save/load, planners, and resharding.
- [PyTorch asynchronous checkpoint recipe](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html): staging, asynchronous writes, and memory considerations.
- [Young, A First Order Approximation to the Optimum Checkpoint Interval](https://doi.org/10.1145/361147.361115): checkpoint interval model.
- [Daly, A Higher Order Estimate of the Optimum Checkpoint Interval for Restart Dumps](https://doi.org/10.1016/j.future.2004.11.016): refined restart interval analysis.
