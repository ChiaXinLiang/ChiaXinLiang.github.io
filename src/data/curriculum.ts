export const LEVELS = [{"id": "beginner", "name": "Beginner", "description": "Build the vocabulary and intuition with explicit assumptions, small examples, and explained equations."}, {"id": "intermediate", "name": "Intermediate", "description": "Connect methods to implementation, measurement, and practical engineering choices."}, {"id": "advanced", "name": "Advanced", "description": "Analyze interacting optimizations, derive tradeoffs, and investigate scaling and failure modes."}] as const;
export const TAGS = [{"name": "AI Infrastructure", "description": "Hardware capacity, networks, kernels, training, and production serving."}, {"name": "Models & Mathematics", "description": "The theory and computational structure behind language models."}, {"name": "Hardware & Co-Design", "description": "Processors, instruction sets, accelerators, and algorithm\u2013hardware tradeoffs."}] as const;
export const PLANNED = [
  {
    "series": "ai-performance",
    "code": "bench-1",
    "title": "Performance Experiments: Baselines, Warmup, Variance, and Reproduction",
    "topic": "Performance Methodology",
    "level": "beginner",
    "status": "idea",
    "order": 6
  },
  {
    "series": "ai-performance",
    "code": "bench-2",
    "title": "Performance Regression CI: Keeping a Speedup After the Next Commit",
    "topic": "Performance Methodology",
    "level": "advanced",
    "status": "idea",
    "order": 15
  },
  {
    "series": "ai-performance",
    "code": "infra-1",
    "title": "NUMA Tuning: CPU Affinity, Memory Placement, and IRQ Locality",
    "topic": "Cluster Infrastructure",
    "level": "intermediate",
    "status": "idea",
    "order": 12
  },
  {
    "series": "ai-performance",
    "code": "infra-2",
    "title": "GPU Containers: Driver Compatibility, Runtime Libraries, and I/O Paths",
    "topic": "Cluster Infrastructure",
    "level": "intermediate",
    "status": "idea",
    "order": 13
  },
  {
    "series": "ai-performance",
    "code": "infra-3",
    "title": "Kubernetes and Slurm: Topology-Aware Placement and Resource Guarantees",
    "topic": "Cluster Infrastructure",
    "level": "intermediate",
    "status": "idea",
    "order": 14
  },
  {
    "series": "ai-networking",
    "code": "net-2",
    "title": "Collective Communication: Ring, Tree, Reduce-Scatter, and All-Gather",
    "topic": "Collectives, Transport, and Overlap",
    "level": "intermediate",
    "status": "idea",
    "order": 4
  },
  {
    "series": "ai-networking",
    "code": "net-3",
    "title": "Debugging NCCL and RDMA: Verify the Path Before Tuning the Knobs",
    "topic": "Collectives, Transport, and Overlap",
    "level": "intermediate",
    "status": "idea",
    "order": 7
  },
  {
    "series": "ai-performance",
    "code": "data-1",
    "title": "The Training Input Pipeline: Workers, Prefetch, Pinned Memory, and GDS",
    "topic": "Cluster Infrastructure",
    "level": "intermediate",
    "status": "idea",
    "order": 11
  },
  {
    "series": "gpu-performance",
    "code": "cuda-1",
    "title": "CUDA Kernel Foundations: Indexing, Launch Geometry, and Boundary Masks",
    "topic": "GPU Execution and Memory",
    "level": "beginner",
    "status": "idea",
    "order": 5
  },
  {
    "series": "gpu-performance",
    "code": "cuda-2",
    "title": "CUDA Correctness: Memory Errors, Races, Barriers, and Compute Sanitizer",
    "topic": "GPU Execution and Memory",
    "level": "intermediate",
    "status": "idea",
    "order": 12
  },
  {
    "series": "gpu-performance",
    "code": "mem-2",
    "title": "Shared-Memory Bank Conflicts: Padding, Swizzling, and Broadcast",
    "topic": "GPU Execution and Memory",
    "level": "intermediate",
    "status": "idea",
    "order": 14
  },
  {
    "series": "gpu-performance",
    "code": "mem-4",
    "title": "Asynchronous Tile Loading: TMA, Barriers, and Buffer Ownership",
    "topic": "GPU Execution and Memory",
    "level": "advanced",
    "status": "idea",
    "order": 23
  },
  {
    "series": "gpu-performance",
    "code": "ktune-4",
    "title": "Instruction-Level Parallelism: Dependency Chains, Registers, and Spills",
    "topic": "GPU Execution and Memory",
    "level": "advanced",
    "status": "idea",
    "order": 22
  },
  {
    "series": "gpu-performance",
    "code": "ktune-5",
    "title": "Tiled GEMM: From a Correct Kernel to a Measured Memory Model",
    "topic": "Kernel Pipelines and Orchestration",
    "level": "intermediate",
    "status": "idea",
    "order": 13
  },
  {
    "series": "gpu-performance",
    "code": "orch-3",
    "title": "Stream-Ordered Allocation: Memory Pools, Events, and Safe Reuse",
    "topic": "Kernel Pipelines and Orchestration",
    "level": "intermediate",
    "status": "idea",
    "order": 15
  },
  {
    "series": "gpu-performance",
    "code": "orch-4",
    "title": "Persistent Kernels and Atomic Queues: Scheduling Irregular Work",
    "topic": "Kernel Pipelines and Orchestration",
    "level": "advanced",
    "status": "idea",
    "order": 24
  },
  {
    "series": "gpu-performance",
    "code": "pt-3",
    "title": "Compiled Training: AOTAutograd, Saved Tensors, and Fusion Boundaries",
    "topic": "PyTorch and Compilers",
    "level": "advanced",
    "status": "idea",
    "order": 25
  },
  {
    "series": "gpu-performance",
    "code": "pt-4",
    "title": "Dynamic Shapes: Guards, Buckets, Padding, and Compilation Cost",
    "topic": "PyTorch and Compilers",
    "level": "intermediate",
    "status": "idea",
    "order": 16
  },
  {
    "series": "gpu-performance",
    "code": "triton-1",
    "title": "Triton Kernel Foundations: Programs, Masks, Layouts, and Correctness",
    "topic": "PyTorch and Compilers",
    "level": "beginner",
    "status": "idea",
    "order": 6
  },
  {
    "series": "gpu-performance",
    "code": "triton-2",
    "title": "Triton Fusion: A Reduction or Normalization Kernel From First Principles",
    "topic": "PyTorch and Compilers",
    "level": "intermediate",
    "status": "idea",
    "order": 17
  },
  {
    "series": "distributed-training",
    "code": "train-1",
    "title": "Training Memory and Step Time: Parameters, Gradients, Optimizer, and Activations",
    "topic": "Distributed Training",
    "level": "beginner",
    "status": "idea",
    "order": 1
  },
  {
    "series": "distributed-training",
    "code": "train-2",
    "title": "DDP: Gradient Buckets and the Backward Communication Timeline",
    "topic": "Distributed Training",
    "level": "intermediate",
    "status": "idea",
    "order": 2
  },
  {
    "series": "distributed-training",
    "code": "train-3",
    "title": "FSDP and ZeRO: What Gets Sharded and What Must Be Materialized",
    "topic": "Distributed Training",
    "level": "intermediate",
    "status": "idea",
    "order": 3
  },
  {
    "series": "distributed-training",
    "code": "train-4",
    "title": "Activation Checkpointing: Selective Recomputation and the Memory–Time Tradeoff",
    "topic": "Distributed Training",
    "level": "intermediate",
    "status": "idea",
    "order": 4
  },
  {
    "series": "distributed-training",
    "code": "train-8",
    "title": "Distributed Checkpoints and Recovery: Goodput Under Failure",
    "topic": "Distributed Training",
    "level": "advanced",
    "status": "idea",
    "order": 8
  },
  {
    "series": "llm-serving",
    "code": "attention-1",
    "title": "FlashAttention: Online Softmax, Exact Tiling, and the I/O Model",
    "topic": "Inference Methods",
    "level": "intermediate",
    "status": "idea",
    "order": 10
  },
  {
    "series": "llm-serving",
    "code": "serve-5",
    "title": "Structured Output: Grammar Masks and Constrained-Decoding Throughput",
    "topic": "Inference Methods",
    "level": "intermediate",
    "status": "idea",
    "order": 11
  },
  {
    "series": "llm-serving",
    "code": "serve-6",
    "title": "Admission Control and QoS: Deadlines, Fairness, and Overload",
    "topic": "Production Serving",
    "level": "advanced",
    "status": "idea",
    "order": 27
  },
  {
    "series": "llm-serving",
    "code": "serve-8",
    "title": "Production Observability: Request Traces, GPU Counters, and Capacity Alerts",
    "topic": "Production Serving",
    "level": "intermediate",
    "status": "idea",
    "order": 12
  },
  {
    "series": "llm-serving",
    "code": "energy-1",
    "title": "Energy per Useful Token: Power Caps, Clocks, Thermals, and SLOs",
    "topic": "Cost and Energy",
    "level": "advanced",
    "status": "idea",
    "order": 26
  },
  {
    "series": "ai-networking",
    "code": "network-1",
    "title": "Bandwidth, Latency, and the Communication Cost Model",
    "topic": "Network Foundations",
    "level": "beginner",
    "status": "idea",
    "order": 1
  },
  {
    "series": "ai-networking",
    "code": "network-2",
    "title": "AI Cluster Topology: PCIe, NVLink, NVSwitch, and Scale-Out Fabrics",
    "topic": "Network Foundations",
    "level": "beginner",
    "status": "idea",
    "order": 2
  },
  {
    "series": "ai-networking",
    "code": "network-3",
    "title": "Ethernet, InfiniBand, and RDMA: The Data Path and Its Guarantees",
    "topic": "Network Foundations",
    "level": "beginner",
    "status": "idea",
    "order": 3
  },
  {
    "series": "ai-networking",
    "code": "network-4",
    "title": "GPUDirect RDMA: GPU–NIC Locality and Transport Verification",
    "topic": "Collectives, Transport, and Overlap",
    "level": "intermediate",
    "status": "idea",
    "order": 5
  },
  {
    "series": "ai-networking",
    "code": "network-5",
    "title": "Congestion and RoCE: ECN, PFC, and Tail Latency",
    "topic": "Congestion, Expert Dispatch, and Reliability",
    "level": "advanced",
    "status": "idea",
    "order": 9
  },
  {
    "series": "ai-networking",
    "code": "network-6",
    "title": "All-to-All for MoE: Expert Dispatch, Imbalance, and Communication Cost",
    "topic": "Congestion, Expert Dispatch, and Reliability",
    "level": "advanced",
    "status": "idea",
    "order": 10
  },
  {
    "series": "ai-networking",
    "code": "network-7",
    "title": "Network Benchmarking: Latency, Bus Bandwidth, and Multi-Node Scaling",
    "topic": "Collectives, Transport, and Overlap",
    "level": "intermediate",
    "status": "idea",
    "order": 6
  },
  {
    "series": "ai-networking",
    "code": "network-8",
    "title": "Network Failures and Stragglers: Diagnosing Distributed Job Stalls",
    "topic": "Congestion, Expert Dispatch, and Reliability",
    "level": "advanced",
    "status": "idea",
    "order": 11
  },
  {
    "series": "distributed-training",
    "code": "train-5",
    "title": "Tensor and Pipeline Parallelism: Partitions, Bubbles, and the Network",
    "topic": "Parallelism Strategies",
    "level": "intermediate",
    "status": "idea",
    "order": 5
  },
  {
    "series": "distributed-training",
    "code": "train-6",
    "title": "Expert, Context, and Sequence Parallelism: Choosing a Process Mesh",
    "topic": "Parallelism Strategies",
    "level": "advanced",
    "status": "idea",
    "order": 6
  },
  {
    "series": "distributed-training",
    "code": "train-7",
    "title": "Training Offload: CPU, NVMe, Bandwidth, and the Critical Path",
    "topic": "Parallelism Strategies",
    "level": "advanced",
    "status": "idea",
    "order": 7
  },
  {
    "series": "llm-architectures",
    "code": "attn-2",
    "title": "Attention state 2: MLA and latent cache reconstruction",
    "topic": "01-building-blocks",
    "level": "advanced",
    "status": "idea",
    "order": 6
  },
  {
    "series": "llm-architectures",
    "code": "compare-1",
    "title": "Comparing LLM architectures: evidence, tradeoffs, and missing disclosures",
    "topic": "04-comparison",
    "level": "advanced",
    "status": "idea",
    "order": 7
  },
  {
    "series": "llm-architectures",
    "code": "ds-1",
    "title": "DeepSeek-V4.1-Flash 1: causal encoder-decoder and phase-specific work",
    "topic": "02-model-case-studies",
    "level": "advanced",
    "status": "idea",
    "order": 8
  },
  {
    "series": "llm-architectures",
    "code": "ds-2",
    "title": "DeepSeek-V4.1-Flash 2: CSA2 sharing and hierarchical sparse indexing",
    "topic": "02-model-case-studies",
    "level": "advanced",
    "status": "idea",
    "order": 9
  },
  {
    "series": "llm-architectures",
    "code": "ds-3",
    "title": "DeepSeek-V4.1-Flash 3: FP4 cache and bounded replay",
    "topic": "02-model-case-studies",
    "level": "advanced",
    "status": "idea",
    "order": 10
  },
  {
    "series": "llm-architectures",
    "code": "model-2",
    "title": "Mixture of experts 1: routing and selected computation",
    "topic": "01-building-blocks",
    "level": "intermediate",
    "status": "idea",
    "order": 3
  },
  {
    "series": "llm-architectures",
    "code": "model-3",
    "title": "Attention state 1: MHA, MQA, and GQA",
    "topic": "01-building-blocks",
    "level": "beginner",
    "status": "idea",
    "order": 2
  },
  {
    "series": "llm-architectures",
    "code": "model-4",
    "title": "Hybrid attention: combining recurrent state with token attention",
    "topic": "01-building-blocks",
    "level": "intermediate",
    "status": "idea",
    "order": 4
  },
  {
    "series": "llm-architectures",
    "code": "moe-2",
    "title": "Mixture of experts 2: load balance, capacity, and dispatch",
    "topic": "01-building-blocks",
    "level": "advanced",
    "status": "idea",
    "order": 11
  },
  {
    "series": "llm-architectures",
    "code": "multi-1",
    "title": "Multimodal LLMs: how image and audio representations meet language",
    "topic": "03-training-and-reasoning",
    "level": "intermediate",
    "status": "idea",
    "order": 5
  },
  {
    "series": "llm-architectures",
    "code": "oss-1",
    "title": "gpt-oss 1: residual structure and sparse expert computation",
    "topic": "02-model-case-studies",
    "level": "advanced",
    "status": "idea",
    "order": 12
  },
  {
    "series": "llm-architectures",
    "code": "oss-2",
    "title": "gpt-oss 2: attention patterns, sinks, and numerical representation",
    "topic": "02-model-case-studies",
    "level": "advanced",
    "status": "idea",
    "order": 13
  },
  {
    "series": "llm-architectures",
    "code": "qwen-1",
    "title": "Qwen3.6: reading a hybrid multimodal model configuration",
    "topic": "02-model-case-studies",
    "level": "advanced",
    "status": "idea",
    "order": 14
  },
  {
    "series": "llm-architectures",
    "code": "reason-1",
    "title": "Reasoning models: architecture, post-training, and inference-time compute",
    "topic": "03-training-and-reasoning",
    "level": "advanced",
    "status": "idea",
    "order": 15
  }
] as const;
