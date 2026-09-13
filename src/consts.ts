// Global site data. Import from anywhere with the `import` keyword.

export const SITE_TITLE = "Marcus's ML Notes";
export const SITE_DESCRIPTION =
	'LLM fundamentals and AI systems performance engineering — notes, deep dives, and back-of-envelope math.';

export const GITHUB_URL = 'https://github.com/ChiaXinLiang';
export const LINKEDIN_URL = 'https://www.linkedin.com/';

export const SERIES = [
  {
    "id": "ai-performance",
    "name": "AI Infrastructure Foundations",
    "tagline": "Measure performance, plan capacity, and understand the host, storage, and cluster beneath a model.",
    "level": "Beginner",
    "tag": "AI Infrastructure"
  },
  {
    "id": "ai-networking",
    "name": "Networking for AI Systems",
    "tagline": "From bandwidth and latency to RDMA, NCCL, topology, congestion, and communication overlap.",
    "level": "Intermediate",
    "tag": "AI Infrastructure"
  },
  {
    "id": "gpu-performance",
    "name": "GPU Programming & Performance",
    "tagline": "Build correct CUDA and Triton kernels, reason about memory traffic, and measure compiler and orchestration tradeoffs.",
    "level": "Advanced",
    "tag": "AI Infrastructure"
  },
  {
    "id": "distributed-training",
    "name": "Distributed Training",
    "tagline": "Understand training state, gradient synchronization, sharding, parallelism, and recovery.",
    "level": "Advanced",
    "tag": "AI Infrastructure"
  },
  {
    "id": "llm-serving",
    "name": "LLM Inference & Serving",
    "tagline": "Connect attention and KV caches to batching, scheduling, production reliability, and useful tokens per dollar.",
    "level": "Advanced",
    "tag": "AI Infrastructure"
  },
  {
    "id": "llm-basics",
    "name": "LLM Foundations & Mathematics",
    "tagline": "Neural networks, probability, Transformers, likelihood, priors, and how language models learn.",
    "level": "Beginner",
    "tag": "Models & Mathematics"
  },
  {
    "id": "llm-architectures",
    "name": "Modern LLM Architectures",
    "tagline": "Read publicly documented model structures: sparse experts, attention state, hybrid blocks, and multimodal computation.",
    "level": "Intermediate",
    "tag": "Models & Mathematics"
  },
  {
    "id": "comp-arch",
    "name": "Computer Architecture & ASIC",
    "tagline": "Processors, memory, Arm, RISC-V, parallel execution, and the design of custom silicon.",
    "level": "Beginner",
    "tag": "Hardware & Co-Design"
  },
  {
    "id": "efficient-ai",
    "name": "Efficient AI & Co-Design",
    "tagline": "Number formats, accelerators, power, and the joint evolution of model algorithms and hardware.",
    "level": "Intermediate",
    "tag": "Hardware & Co-Design"
  }
] as const;
export type SeriesId = (typeof SERIES)[number]["id"];
