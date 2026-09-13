# Article visual review and walkthrough

Reviewed 508 image references across all 171 technical articles in 11 published series: 337 inline figures and 171 covers. Replaced 63 generic overview figures with original concept illustrations, updated 149 covers (63 illustrated overviews plus 86 existing mechanism figures), retained 274 useful supporting figures and 22 existing concept covers. Fixed the broken RNN cover.

Illustrations use recognizable hardware, tensors, caches, distributions, queues, and spatial relationships where appropriate. Precise plots, bit layouts, timelines, and worked numerical examples remain beside the article text. The compositions vary by concept rather than repeating four text cards. They are original illustrations inspired by clear system-design teaching, without copied artwork, promotional branding, or invented benchmark results.

All newly generated bitmaps use the built-in image_gen tool. See [prompt records](./overview-illustration-prompts.json) and [complete image decisions and saved asset paths](./overview-image-audit.json). Initial prompts, nine main corrections, and the final asynchronous-loading correction are recorded verbatim; four additional correction briefs and the approved network brief are identified as briefs.

Technical review corrected joint MLA cache state, posterior-mode placement, FP4 scale formats, KV append/paging paths, FlashAttention normalization, structured-output renormalization, Triton variance, collective bucket matching, GPU barrier participation, learned attention sinks, benchmark interpretation, persistent-worker draining, and producer/consumer buffer ownership.

## Shared visual presentation

Every article uses the shared visual walkthrough: overview first, followed by its existing deep-dive diagrams in reading order. The top carousel supports numbered selection, bounded previous/next controls, keyboard arrows, and horizontal swipe. Article images remain in their sections and can be opened by mouse or keyboard. The modal viewer supports navigation, fit/zoom, copy as PNG, download of the served image, Escape/backdrop close, focus restoration, and scroll locking. Copy requires a browser supporting image clipboard writes on HTTPS or localhost; download remains available. Articles without body figures use their cover; future articles inherit the same behavior.

Added 77 focused deep-dive illustrations to articles that previously had only an overview. All 171 technical articles now have 2–4 teaching figures (414 inline figures total). Each new figure is placed in the section it explains; existing detailed diagrams remain. See [deep-dive prompts](./deep-dive-illustration-prompts.json) and [saved deep-dive assets](./deep-dive-illustration-assets.json). Future section images join the walkthrough automatically.

## New overview assets

| Article | Saved overview |
| --- | --- |
| FSDP and ZeRO: What Gets Sharded and What Must Be Materialized | [`src/content/blog/fsdp-zero-sharding-materialization/section-overview.png`](../src/content/blog/fsdp-zero-sharding-materialization/section-overview.png) |
| Instruction-Level Parallelism: Dependency Chains, Registers, and Spills | [`src/content/blog/instruction-level-parallelism-registers-spills/section-overview.png`](../src/content/blog/instruction-level-parallelism-registers-spills/section-overview.png) |
| Congestion and RoCE: ECN, PFC, and Tail Latency | [`src/content/blog/roce-congestion-ecn-pfc-tail-latency/section-overview.png`](../src/content/blog/roce-congestion-ecn-pfc-tail-latency/section-overview.png) |
| Activation Checkpointing: Selective Recomputation and the Memory–Time Tradeoff | [`src/content/blog/activation-checkpointing-selective-recomputation/section-overview.png`](../src/content/blog/activation-checkpointing-selective-recomputation/section-overview.png) |
| Ethernet, InfiniBand, and RDMA: The Data Path and Its Guarantees | [`src/content/blog/ethernet-infiniband-rdma-data-path-guarantees/section-overview.png`](../src/content/blog/ethernet-infiniband-rdma-data-path-guarantees/section-overview.png) |
| The Training Input Pipeline: Workers, Prefetch, Pinned Memory, and GDS | [`src/content/blog/training-input-pipeline-workers-pinned-memory-gds/section-overview.png`](../src/content/blog/training-input-pipeline-workers-pinned-memory-gds/section-overview.png) |
| Hybrid Attention: Combining Recurrent State with Token Attention | [`src/content/blog/hybrid-attention-recurrent-state-token-history/section-overview.png`](../src/content/blog/hybrid-attention-recurrent-state-token-history/section-overview.png) |
| GPUDirect RDMA: GPU–NIC Locality and Transport Verification | [`src/content/blog/gpudirect-rdma-locality-transport-verification/section-overview.png`](../src/content/blog/gpudirect-rdma-locality-transport-verification/section-overview.png) |
| All-to-All for MoE: Expert Dispatch, Imbalance, and Communication Cost | [`src/content/blog/moe-all-to-all-dispatch-imbalance-cost/section-overview.png`](../src/content/blog/moe-all-to-all-dispatch-imbalance-cost/section-overview.png) |
| CUDA Correctness: Memory Errors, Races, Barriers, and Compute Sanitizer | [`src/content/blog/cuda-correctness-memory-races-barriers-sanitizer/section-overview.png`](../src/content/blog/cuda-correctness-memory-races-barriers-sanitizer/section-overview.png) |
| Training Offload: CPU, NVMe, Bandwidth, and the Critical Path | [`src/content/blog/training-offload-cpu-nvme-critical-path/section-overview.png`](../src/content/blog/training-offload-cpu-nvme-critical-path/section-overview.png) |
| Mixture of Experts 1: Routing and Selected Computation | [`src/content/blog/mixture-of-experts-routing-selected-computation/section-overview.png`](../src/content/blog/mixture-of-experts-routing-selected-computation/section-overview.png) |
| Serving Observability: Request Traces, GPU Counters, and Capacity Alerts | [`src/content/blog/serving-observability-traces-counters-capacity/section-overview.png`](../src/content/blog/serving-observability-traces-counters-capacity/section-overview.png) |
| DeepSeek-V4.1-Flash 3: FP4 Cache and Bounded Replay | [`src/content/blog/deepseek-v41-flash-fp4-cache-bounded-replay/section-overview.png`](../src/content/blog/deepseek-v41-flash-fp4-cache-bounded-replay/section-overview.png) |
| Expert, Context, and Sequence Parallelism: Choosing a Process Mesh | [`src/content/blog/expert-context-sequence-parallel-process-mesh/section-overview.png`](../src/content/blog/expert-context-sequence-parallel-process-mesh/section-overview.png) |
| FlashAttention: Online Softmax, Exact Tiling, and the I/O Model | [`src/content/blog/flashattention-online-softmax-io-model/section-overview.png`](../src/content/blog/flashattention-online-softmax-io-model/section-overview.png) |
| Energy per Useful Token: Power Caps, Clocks, Thermals, and SLOs | [`src/content/blog/energy-useful-token-power-clocks-slos/section-overview.png`](../src/content/blog/energy-useful-token-power-clocks-slos/section-overview.png) |
| Triton Kernel Foundations: Programs, Masks, Layouts, and Correctness | [`src/content/blog/triton-programs-masks-layout-correctness/section-overview.png`](../src/content/blog/triton-programs-masks-layout-correctness/section-overview.png) |
| GPU Containers: Driver Compatibility, Runtime Libraries, and I/O Paths | [`src/content/blog/gpu-containers-compatibility-libraries-io-paths/section-overview.png`](../src/content/blog/gpu-containers-compatibility-libraries-io-paths/section-overview.png) |
| DeepSeek-V4.1-Flash 1: Causal Encoder-Decoder and Phase-Specific Work | [`src/content/blog/deepseek-v41-flash-causal-encoder-decoder/section-overview.png`](../src/content/blog/deepseek-v41-flash-causal-encoder-decoder/section-overview.png) |
| gpt-oss 1: Residual Structure and Sparse Expert Computation | [`src/content/blog/gpt-oss-residual-structure-sparse-experts/section-overview.png`](../src/content/blog/gpt-oss-residual-structure-sparse-experts/section-overview.png) |
| Structured Output: Grammar Masks and Constrained Decoding Throughput | [`src/content/blog/structured-output-grammar-masks-throughput/section-overview.png`](../src/content/blog/structured-output-grammar-masks-throughput/section-overview.png) |
| Admission Control and QoS: Deadlines, Fairness, and Overload | [`src/content/blog/admission-control-qos-deadlines-fairness/section-overview.png`](../src/content/blog/admission-control-qos-deadlines-fairness/section-overview.png) |
| NUMA Tuning: CPU Affinity, Memory Placement, and IRQ Locality | [`src/content/blog/numa-cpu-memory-irq-locality/section-overview.png`](../src/content/blog/numa-cpu-memory-irq-locality/section-overview.png) |
| Mixture of Experts 2: Load Balance, Capacity, and Dispatch | [`src/content/blog/mixture-of-experts-capacity-load-balance-dispatch/section-overview.png`](../src/content/blog/mixture-of-experts-capacity-load-balance-dispatch/section-overview.png) |
| Shared-Memory Bank Conflicts: Padding, Swizzling, and Broadcast | [`src/content/blog/shared-memory-banks-padding-swizzling-broadcast/section-overview.png`](../src/content/blog/shared-memory-banks-padding-swizzling-broadcast/section-overview.png) |
| Debugging NCCL and RDMA: Verify the Path Before Tuning the Knobs | [`src/content/blog/debugging-nccl-rdma-verify-path-before-tuning/section-overview.png`](../src/content/blog/debugging-nccl-rdma-verify-path-before-tuning/section-overview.png) |
| Distributed Checkpoints and Recovery: Goodput Under Failure | [`src/content/blog/distributed-checkpoints-recovery-goodput/section-overview.png`](../src/content/blog/distributed-checkpoints-recovery-goodput/section-overview.png) |
| Network Failures and Stragglers: Diagnosing Distributed Job Stalls | [`src/content/blog/network-failures-stragglers-distributed-stalls/section-overview.png`](../src/content/blog/network-failures-stragglers-distributed-stalls/section-overview.png) |
| Multimodal LLMs: How Image and Audio Representations Meet Language | [`src/content/blog/multimodal-llms-image-audio-language-representations/section-overview.png`](../src/content/blog/multimodal-llms-image-audio-language-representations/section-overview.png) |
| CUDA Kernel Foundations: Indexing, Launch Geometry, and Boundary Masks | [`src/content/blog/cuda-kernel-indexing-launch-boundaries/section-overview.png`](../src/content/blog/cuda-kernel-indexing-launch-boundaries/section-overview.png) |
| Reasoning Models: Architecture, Post-Training, and Inference-Time Compute | [`src/content/blog/reasoning-models-post-training-inference-compute/section-overview.png`](../src/content/blog/reasoning-models-post-training-inference-compute/section-overview.png) |
| Triton Fusion: A Reduction or Normalization Kernel From First Principles | [`src/content/blog/triton-fused-normalization-reduction-first-principles/section-overview.png`](../src/content/blog/triton-fused-normalization-reduction-first-principles/section-overview.png) |
| Training Memory and Step Time: Account for Every State | [`src/content/blog/training-memory-and-step-time/section-overview.png`](../src/content/blog/training-memory-and-step-time/section-overview.png) |
| Network Benchmarking: Latency, Bus Bandwidth, and Multi-Node Scaling | [`src/content/blog/network-benchmarking-latency-bus-bandwidth-scaling/section-overview.png`](../src/content/blog/network-benchmarking-latency-bus-bandwidth-scaling/section-overview.png) |
| Kubernetes and Slurm: Topology-Aware Placement and Resource Guarantees | [`src/content/blog/kubernetes-slurm-topology-resource-guarantees/section-overview.png`](../src/content/blog/kubernetes-slurm-topology-resource-guarantees/section-overview.png) |
| Tiled GEMM: From a Correct Kernel to a Measured Memory Model | [`src/content/blog/tiled-gemm-correctness-measured-memory-model/section-overview.png`](../src/content/blog/tiled-gemm-correctness-measured-memory-model/section-overview.png) |
| DDP: Gradient Buckets and the Backward Communication Timeline | [`src/content/blog/ddp-gradient-buckets-backward-timeline/section-overview.png`](../src/content/blog/ddp-gradient-buckets-backward-timeline/section-overview.png) |
| DeepSeek-V4.1-Flash 2: CSA2 Sharing and Hierarchical Sparse Indexing | [`src/content/blog/deepseek-v41-flash-csa2-hierarchical-indexing/section-overview.png`](../src/content/blog/deepseek-v41-flash-csa2-hierarchical-indexing/section-overview.png) |
| Dynamic Shapes: Guards, Buckets, Padding, and Compilation Cost | [`src/content/blog/dynamic-shapes-guards-buckets-compilation/section-overview.png`](../src/content/blog/dynamic-shapes-guards-buckets-compilation/section-overview.png) |
| Tensor and Pipeline Parallelism: Partitions, Bubbles, and the Network | [`src/content/blog/tensor-pipeline-parallelism-partitions-bubbles/section-overview.png`](../src/content/blog/tensor-pipeline-parallelism-partitions-bubbles/section-overview.png) |
| Collective Communication: Ring, Tree, Reduce-Scatter, and All-Gather | [`src/content/blog/collectives-ring-tree-reduce-scatter-all-gather/section-overview.png`](../src/content/blog/collectives-ring-tree-reduce-scatter-all-gather/section-overview.png) |
| Asynchronous Tile Loading: TMA, Barriers, and Buffer Ownership | [`src/content/blog/asynchronous-tile-loading-tma-buffer-ownership/section-overview.png`](../src/content/blog/asynchronous-tile-loading-tma-buffer-ownership/section-overview.png) |
| Comparing LLM Architectures: Evidence, Tradeoffs, and Missing Disclosures | [`src/content/blog/comparing-llm-architectures-evidence-tradeoffs/section-overview.png`](../src/content/blog/comparing-llm-architectures-evidence-tradeoffs/section-overview.png) |
| Qwen3.6: Reading a Hybrid Multimodal Model Configuration | [`src/content/blog/qwen36-hybrid-multimodal-configuration/section-overview.png`](../src/content/blog/qwen36-hybrid-multimodal-configuration/section-overview.png) |
| Stream-Ordered Allocation: Memory Pools, Events, and Safe Reuse | [`src/content/blog/stream-ordered-allocation-pools-events-reuse/section-overview.png`](../src/content/blog/stream-ordered-allocation-pools-events-reuse/section-overview.png) |
| gpt-oss 2: Attention Patterns, Sinks, and Numerical Representation | [`src/content/blog/gpt-oss-attention-patterns-sinks-representation/section-overview.png`](../src/content/blog/gpt-oss-attention-patterns-sinks-representation/section-overview.png) |
| AI Cluster Topology: PCIe, NVLink, NVSwitch, and Scale-Out Fabrics | [`src/content/blog/ai-cluster-topology-local-and-scale-out/section-overview.png`](../src/content/blog/ai-cluster-topology-local-and-scale-out/section-overview.png) |
| Persistent Kernels and Atomic Queues: Scheduling Irregular Work | [`src/content/blog/persistent-kernels-atomic-queues-irregular-work/section-overview.png`](../src/content/blog/persistent-kernels-atomic-queues-irregular-work/section-overview.png) |
| Performance Regression CI: Keeping a Speedup After the Next Commit | [`src/content/blog/performance-regression-ci-reproducible-speedups/section-overview.png`](../src/content/blog/performance-regression-ci-reproducible-speedups/section-overview.png) |
| Performance Experiments: Baselines, Warmup, Variance, and Reproduction | [`src/content/blog/performance-experiments-warmup-variance-reproduction/section-overview.png`](../src/content/blog/performance-experiments-warmup-variance-reproduction/section-overview.png) |
| Compiled Training: AOTAutograd, Saved Tensors, and Fusion Boundaries | [`src/content/blog/compiled-training-aotautograd-saved-tensors-fusion/section-overview.png`](../src/content/blog/compiled-training-aotautograd-saved-tensors-fusion/section-overview.png) |
| Attention State 2: MLA and Latent Cache Reconstruction | [`src/content/blog/attention-state-mla-latent-cache-reconstruction/section-overview.png`](../src/content/blog/attention-state-mla-latent-cache-reconstruction/section-overview.png) |
| Attention State 1: MHA, MQA, and GQA | [`src/content/blog/attention-state-mha-mqa-gqa/section-overview.png`](../src/content/blog/attention-state-mha-mqa-gqa/section-overview.png) |
| Out-of-Order Execution: The Illusion of Sequential Code | [`src/content/blog/out-of-order-execution/section-overview.png`](../src/content/blog/out-of-order-execution/section-overview.png) |
| MAP Estimation: How a Prior Changes What a Model Learns | [`src/content/blog/map-estimation-and-priors/section-overview.png`](../src/content/blog/map-estimation-and-priors/section-overview.png) |
| Maximum Likelihood Estimation: From Observed Data to a Training Objective | [`src/content/blog/maximum-likelihood-estimation/section-overview.png`](../src/content/blog/maximum-likelihood-estimation/section-overview.png) |
| Assembly Lines Inside a GPU: Warp Specialization | [`src/content/blog/warp-specialization/section-overview.png`](../src/content/blog/warp-specialization/section-overview.png) |
| NVFP4 vs MXFP4: Inside the 4-Bit Format War | [`src/content/blog/nvfp4-vs-mxfp4-the-4bit-format-war/section-overview.png`](../src/content/blog/nvfp4-vs-mxfp4-the-4bit-format-war/section-overview.png) |
| Reading a Modern LLM: From Model Configuration to Computational Structure | [`src/content/blog/reading-a-modern-llm/section-overview.png`](../src/content/blog/reading-a-modern-llm/section-overview.png) |
| The KV Cache, Explained for Engineers | [`src/content/blog/kv-cache-explained/section-overview.png`](../src/content/blog/kv-cache-explained/section-overview.png) |
| Profiling Basics: Finding Where the Time Actually Goes | [`src/content/blog/profiling-basics-where-time-goes/section-overview.png`](../src/content/blog/profiling-basics-where-time-goes/section-overview.png) |
| Bandwidth, Latency, and the Communication Cost Model | [`src/content/blog/bandwidth-latency-communication-cost-model/section-overview.png`](../src/content/blog/bandwidth-latency-communication-cost-model/section-overview.png) |

## Reused cover figures

| Article | Existing concept figure |
| --- | --- |
| a-year-of-kernelbench | ./loop.png |
| ai-optimizing-ai | ./pattern.png |
| arithmetic-intensity | ./fig-roofline.png |
| arm-aarch64-registers-and-ecosystem | ./figure-01.png |
| arm-riscv-x86-comparing-without-myths | ./figure-01.png |
| asic-vs-fpga-vs-gpu | ./spectrum.png |
| attention-in-plain-words | ./attention-lines.png |
| batching-the-biggest-throughput-lever | ./gemv-vs-gemm.png |
| benchmarking-pitfalls | ./goodput-slo.png |
| blackwell-to-rubin-memory-math | ./capacity-vs-bandwidth.png |
| branch-prediction-the-cpu-gambler | ./pipeline-flush.png |
| caches-how-locality-rescues-speed | ./cache-line.png |
| case-first-token-takes-8-seconds | ./prefill-flops.png |
| case-gpu-busy-tokens-low | figure-02.png |
| case-latency-spikes-batch-scheduler | figure-02.png |
| case-long-chats-get-slower | figure-02.png |
| case-p50-great-p99-terrible | figure-02.png |
| case-quantized-model-isnt-faster | ./fused-vs-fallback.png |
| case-slower-on-the-bigger-gpu | ./tp2-pcie.png |
| case-throughput-collapses-at-30-users | ./fig-cliff.png |
| cloud-gpu-price-to-cost-per-million-tokens | ./figure-02.png |
| cnn-how-machines-learned-to-see | ./convolution.png |
| compute-bound-vs-memory-bound | ./roofline.png |
| cost-per-million-tokens | ./cost-ladder.png |
| cpu-vs-gpu-latency-vs-throughput-machines | ./die-budget.png |
| cross-entropy-and-kl-divergence | figure-02.png |
| cuda-graphs-record-once-replay-forever | ./launch-timeline.png |
| does-llama-70b-fit-on-one-h100 | ./figure-01.png |
| every-kernel-is-one-of-four-things | ./fig-quadrant.png |
| free-speed-from-the-os | ./numa-path.png |
| from-dram-to-hbm | ./inside-an-hbm-stack.png |
| generalization-and-regularization | figure-02.png |
| goodput-vs-utilization | ./goodput-gap.png |
| gpu-economics-from-openais-price-sheet | ./price-ratios.png |
| gpu-memory-math-will-it-fit | ./kv-cache-anatomy.png |
| gpus-waiting-on-storage | ./data-path.png |
| hide-the-network-overlap-communication | ./overlap-timeline.png |
| how-an-llm-generates-text | ./prefill-decode.png |
| how-big-a-batch-before-compute-bound | ./figure-01.png |
| how-models-learn | ./gradient-descent.png |
| how-much-kv-cache-for-128k-context | ./figure-01.png |
| hyperscaler-inference-silicon | ./three-chips.png |
| inference-engines-that-retune-themselves | ./step-bytes.png |
| inside-torch-compile-graph-breaks | ./compile-pipeline.png |
| instruction-sets-software-hardware-contract | ./figure-02.png |
| kv-cache-first-class-citizen | ./prefix-pool.png |
| latency-vs-throughput | ./batching-lever.png |
| memory-coalescing | ./fig-coalescing.png |
| nvl72-one-rack-one-giant-gpu | ./inside-the-rack.png |
| occupancy-and-the-roofline | ./latency-hiding.png |
| prefill-gets-its-own-chip-rubin-cpx | ./roofline.png |
| pretraining-finetuning-rlhf | ./rlhf-loop.png |
| probability-for-machine-learning | figure-01.png |
| quantization-what-you-gain-what-you-lose | ./quant-targets.png |
| riscv-small-base-extensible-system | ./figure-01.png |
| rnn-lstm-and-the-wall | ./rnn-unrolled.png |
| rtl-to-gdsii-in-plain-words | ./flow.png |
| serving-moe-giants | ./fig-parallelism-axes.png |
| simd-one-instruction-many-numbers | ./vector-add.png |
| speculative-decoding-in-one-picture | ./spec-one-picture.png |
| stop-serving-prefill-and-decode-together | ./fig-arch.png |
| systolic-arrays-the-idea-inside-tpus | ./fig-dataflow.png |
| the-9-percent-model-qwen3-next | ./fig-two-levers.png |
| the-default-cuda-stream-hidden-barrier | ./default-stream-barrier.png |
| the-economics-of-a-tapeout | ./nre-stack.png |
| the-golden-age-of-domain-specific-chips | ./cpu-vs-dsa.png |
| the-memory-wall-latency-numbers | ./latency-ladder.png |
| the-prefill-decode-disaggregation-story | ./disagg-architecture.png |
| the-wafer-scale-bet-cerebras | ./wafer-uncut.png |
| theoretical-tokens-per-second-from-bandwidth | ./figure-01.png |
| tokenization-why-llms-dont-see-words | ./strawberry-tokens.png |
| tokens-per-megawatt | ./fig-power-gap.png |
| tokens-per-second-what-it-hides | ./anatomy-of-a-request.png |
| torch-compile-and-hidden-syncs | ./fig-async-pipeline.png |
| training-in-4-bit | ./fp4-anatomy.png |
| transformer-architecture-in-one-picture | ./transformer-equation-block.png |
| ttft-and-tpot | ./prefill-decode.png |
| tuning-inference-at-scale | ./fig-chunked-prefill.png |
| vllm-tensorrt-llm-sglang-compared | ./three-designs.png |
| what-a-cpu-actually-does | ./fde-loop.png |
| what-does-an-ml-performance-engineer-do | ./triangle.png |
| what-is-a-neural-network | ./neuron.png |
| when-a-kernel-cuts-api-prices | ./dsa-pipeline.png |
| why-ai-chips-are-the-easy-asics | ./systolic-array.png |
| why-gpus-sit-idle | ./warp-switching.png |
| why-transformers-won | ./serial-vs-parallel.png |

## Verification

The production build completed with 189 pages. All 171 technical articles have rendered mathematics, valid source and built image links, and prose counts within 1,800–2,500 words. Their 414 inline figures comprise 103 articles with 2 figures, 64 with 3, and 4 with 4. Numbering and Previous/Next links passed across all 11 series; topic search and filters passed.

DOM interaction checks passed for all 171 technical article pages: slide selection and boundaries, keyboard navigation, focus restoration, modal/backdrop close, fit/zoom, swipe and synthetic-click suppression, copy/download success and errors, unsupported clipboard, no-image fallback, and repeat initialization. The non-PNG copy path converts SVG using a native canvas; downloads keep the served format. Dialog and clipboard APIs are stubbed in DOM tests; clipboard availability is detected by the actual reader browser. Escape and modal focus containment use the native HTML dialog.

To repeat the interaction checks after a build, install the temporary test utility with `npm install --prefix /tmp/article-visual-tests jsdom --ignore-scripts --no-audit --no-fund`, then run `node tools/check-article-visuals.mjs /tmp/article-visual-tests`. No client framework or new runtime dependency was added.
