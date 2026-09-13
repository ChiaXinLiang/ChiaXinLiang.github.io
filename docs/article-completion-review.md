# Article completion review

29 previously unfinished articles completed on September 13, 2026. 26 editorial records remain unfinished.

Each addition has an original section overview, defined mathematical quantities, worked examples, and primary-source links. Numerical performance examples are identified as illustrative.

| Series | Article | Prose words before publication |
| --- | --- | --- |
| ai-networking | [Collective Communication: Ring, Tree, Reduce-Scatter, and All-Gather](../src/content/blog/collectives-ring-tree-reduce-scatter-all-gather/index.md) | 1808 |
| ai-networking | [Debugging NCCL and RDMA: Verify the Path Before Tuning the Knobs](../src/content/blog/debugging-nccl-rdma-verify-path-before-tuning/index.md) | 1804 |
| ai-networking | [Bandwidth, Latency, and the Communication Cost Model](../src/content/blog/bandwidth-latency-communication-cost-model/index.md) | 1810 |
| ai-networking | [AI Cluster Topology: PCIe, NVLink, NVSwitch, and Scale-Out Fabrics](../src/content/blog/ai-cluster-topology-local-and-scale-out/index.md) | 1803 |
| ai-networking | [Ethernet, InfiniBand, and RDMA: The Data Path and Its Guarantees](../src/content/blog/ethernet-infiniband-rdma-data-path-guarantees/index.md) | 1810 |
| ai-networking | [GPUDirect RDMA: GPU–NIC Locality and Transport Verification](../src/content/blog/gpudirect-rdma-locality-transport-verification/index.md) | 1807 |
| ai-networking | [Congestion and RoCE: ECN, PFC, and Tail Latency](../src/content/blog/roce-congestion-ecn-pfc-tail-latency/index.md) | 1810 |
| ai-networking | [All-to-All for MoE: Expert Dispatch, Imbalance, and Communication Cost](../src/content/blog/moe-all-to-all-dispatch-imbalance-cost/index.md) | 1805 |
| ai-networking | [Network Benchmarking: Latency, Bus Bandwidth, and Multi-Node Scaling](../src/content/blog/network-benchmarking-latency-bus-bandwidth-scaling/index.md) | 1817 |
| ai-networking | [Network Failures and Stragglers: Diagnosing Distributed Job Stalls](../src/content/blog/network-failures-stragglers-distributed-stalls/index.md) | 1813 |
| ai-performance | [Performance Experiments: Baselines, Warmup, Variance, and Reproduction](../src/content/blog/performance-experiments-warmup-variance-reproduction/index.md) | 1804 |
| ai-performance | [Performance Regression CI: Keeping a Speedup After the Next Commit](../src/content/blog/performance-regression-ci-reproducible-speedups/index.md) | 1800 |
| ai-performance | [The Training Input Pipeline: Workers, Prefetch, Pinned Memory, and GDS](../src/content/blog/training-input-pipeline-workers-pinned-memory-gds/index.md) | 1805 |
| ai-performance | [NUMA Tuning: CPU Affinity, Memory Placement, and IRQ Locality](../src/content/blog/numa-cpu-memory-irq-locality/index.md) | 1803 |
| ai-performance | [GPU Containers: Driver Compatibility, Runtime Libraries, and I/O Paths](../src/content/blog/gpu-containers-compatibility-libraries-io-paths/index.md) | 1803 |
| ai-performance | [Kubernetes and Slurm: Topology-Aware Placement and Resource Guarantees](../src/content/blog/kubernetes-slurm-topology-resource-guarantees/index.md) | 1816 |
| distributed-training | [Training Memory and Step Time: Account for Every State](../src/content/blog/training-memory-and-step-time/index.md) | 1827 |
| distributed-training | [DDP: Gradient Buckets and the Backward Communication Timeline](../src/content/blog/ddp-gradient-buckets-backward-timeline/index.md) | 1902 |
| distributed-training | [FSDP and ZeRO: What Gets Sharded and What Must Be Materialized](../src/content/blog/fsdp-zero-sharding-materialization/index.md) | 1905 |
| distributed-training | [Activation Checkpointing: Selective Recomputation and the Memory–Time Tradeoff](../src/content/blog/activation-checkpointing-selective-recomputation/index.md) | 1916 |
| distributed-training | [Tensor and Pipeline Parallelism: Partitions, Bubbles, and the Network](../src/content/blog/tensor-pipeline-parallelism-partitions-bubbles/index.md) | 1814 |
| distributed-training | [Expert, Context, and Sequence Parallelism: Choosing a Process Mesh](../src/content/blog/expert-context-sequence-parallel-process-mesh/index.md) | 1873 |
| distributed-training | [Training Offload: CPU, NVMe, Bandwidth, and the Critical Path](../src/content/blog/training-offload-cpu-nvme-critical-path/index.md) | 1849 |
| distributed-training | [Distributed Checkpoints and Recovery: Goodput Under Failure](../src/content/blog/distributed-checkpoints-recovery-goodput/index.md) | 1809 |
| llm-serving | [FlashAttention: Online Softmax, Exact Tiling, and the I/O Model](../src/content/blog/flashattention-online-softmax-io-model/index.md) | 1855 |
| llm-serving | [Energy per Useful Token: Power Caps, Clocks, Thermals, and SLOs](../src/content/blog/energy-useful-token-power-clocks-slos/index.md) | 1808 |
| llm-serving | [Structured Output: Grammar Masks and Constrained Decoding Throughput](../src/content/blog/structured-output-grammar-masks-throughput/index.md) | 1825 |
| llm-serving | [Admission Control and QoS: Deadlines, Fairness, and Overload](../src/content/blog/admission-control-qos-deadlines-fairness/index.md) | 1828 |
| llm-serving | [Serving Observability: Request Traces, GPU Counters, and Capacity Alerts](../src/content/blog/serving-observability-traces-counters-capacity/index.md) | 1830 |

## Verification

- Static build succeeds with 124 published database records, including the welcome page.
- All 123 long-form articles have 1,800–2,500 prose words, rendered equations, existing image assets, and valid internal routes.
- All 9 series have consecutive displayed article numbers and matching Previous/Next links.
- Roadmap paths, selection, reading progress, persistence, and reset/cancel pass the command-line interaction checks.
- Browser inspection is unavailable because the computer-use service returned an administrator policy verification error. Original figures were rendered locally and a representative figure was visually inspected.
- Moved Megatron source URLs were replaced with working current context-parallel, token-dispatcher, and pipeline-schedule documentation.

The compact roadmap includes Distributed Training. Its connection paths are derived from background relationships rather than a fixed list of SVG coordinates.
