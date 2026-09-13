# Efficient ML expansion publication review

Completed: 2026-09-13.

All 22 approved articles are published: 14 in **Efficient ML: Compression & Model Design**, 6 in **Efficient Vision & Generative Models**, and 2 additional state-space articles in **LLM Architectures**. The site now has 171 technical articles across 11 public series, plus its welcome post. No unfinished editorial entries remain. Every series remains below the 30-article limit.

Each new article contains defined equations, theoretical mechanisms, worked examples, primary references, and an original concept illustration. Figures use different central compositions to explain the article rather than repeating a generic workflow/error layout. Numerical plots and image analogies are explicitly illustrative. The visual reference informed explanatory clarity; its branded artwork was not copied.

## Article inventory

| Article | Prose words | Overview concept |
| --- | ---: | --- |
| [Efficiency Objectives: Accuracy, Latency, Memory, and Pareto Tradeoffs](https://chiaxinliang.github.io/blog/efficiency-objectives-pareto-deployment/) | 1877 | quality–latency feasible region |
| [Pruning 1: Magnitude, Saliency, and Recovery Training](https://chiaxinliang.github.io/blog/pruning-saliency-magnitude-recovery/) | 1877 | retained neural connections and saliency |
| [Pruning 2: Structured Sparsity and Real Hardware Speedups](https://chiaxinliang.github.io/blog/structured-sparsity-hardware-speedups/) | 1833 | four physically different sparsity patterns |
| [Quantization 1: Scales, Clipping, Calibration, and Error](https://chiaxinliang.github.io/blog/quantization-scales-clipping-calibration/) | 1841 | clipping versus quantization-grid resolution |
| [Quantization 2: PTQ, QAT, and Fake-Quantized Training](https://chiaxinliang.github.io/blog/ptq-qat-fake-quantization-training/) | 1885 | shadow parameters and simulated forward quantization |
| [LLM Quantization Methods: GPTQ, AWQ, and SmoothQuant](https://chiaxinliang.github.io/blog/gptq-awq-smoothquant-mechanisms/) | 1872 | three distinct compensation/scaling mechanisms |
| [Mixed Precision: Layer Sensitivity and a Deployment Budget](https://chiaxinliang.github.io/blog/mixed-precision-layer-sensitivity-budget/) | 1863 | precision allocation and payload budget |
| [Distillation 1: Teacher–Student Objectives and Temperature](https://chiaxinliang.github.io/blog/distillation-temperature-teacher-student/) | 1889 | teacher/student soft-probability distributions |
| [Distillation 2: Features, Data, and Deployment Tradeoffs](https://chiaxinliang.github.io/blog/distillation-features-data-deployment/) | 1843 | feature alignment versus relational geometry |
| [LoRA: Low-Rank Updates and Training-State Memory](https://chiaxinliang.github.io/blog/lora-low-rank-updates-memory/) | 1845 | skinny factor matrices beside a frozen base |
| [QLoRA: Quantized Base Weights and Adapter Numerics](https://chiaxinliang.github.io/blog/qlora-quantized-base-adapter-numerics/) | 1870 | nonuniform codes, scale storage, and adapter branch |
| [Hardware-Aware Model Design: Shapes, Latency, and Search Spaces](https://chiaxinliang.github.io/blog/hardware-aware-model-design-latency/) | 1845 | tile spillover and roofline bound |
| [Neural Architecture Search: Objectives, Search Cost, and Evidence](https://chiaxinliang.github.io/blog/neural-architecture-search-cost-evidence/) | 1852 | search choices, shared evaluator, and final subnet |
| [A Reproducible Compression Experiment: Quality and Deployment](https://chiaxinliang.github.io/blog/compression-experiment-quality-deployment/) | 1844 | controlled candidates and acceptance frontier |
| [Vision Transformers: Patches, Positions, and Computational Cost](https://chiaxinliang.github.io/blog/vision-transformer-patches-position-cost/) | 1850 | spatial patch grid, position, and pairwise attention |
| [Efficient Vision: Token Reduction and Resolution Tradeoffs](https://chiaxinliang.github.io/blog/efficient-vision-token-reduction-resolution/) | 1845 | discarded regions versus size-tracked merging |
| [Diffusion Foundations: Noise, Reverse Prediction, and Training](https://chiaxinliang.github.io/blog/diffusion-noise-prediction-training-objective/) | 1829 | synthetic signal progressively obscured by noise |
| [Efficient Diffusion Sampling: Solvers, ODEs, and SDEs](https://chiaxinliang.github.io/blog/diffusion-sampling-solvers-ode-sde/) | 1848 | smooth trajectory versus coarse numerical updates |
| [Diffusion Distillation: Fewer Steps and Quality Tradeoffs](https://chiaxinliang.github.io/blog/diffusion-distillation-fewer-generation-steps/) | 1836 | one student transition versus two teacher transitions |
| [Vision and Generation Deployment: A Controlled Experiment](https://chiaxinliang.github.io/blog/vision-generation-deployment-experiment/) | 1842 | fixed pipeline costs and usable-output quality |
| [State-Space Models: Dynamics, Discretization, and Stability](https://chiaxinliang.github.io/blog/state-space-models-dynamics-discretization/) | 1859 | continuous decay, sampled dynamics, and retained state |
| [Efficient State-Space Execution: Scans, Recurrence, and Hybrids](https://chiaxinliang.github.io/blog/state-space-execution-scans-recurrence-hybrids/) | 1853 | temporal transition-composition tree |

Word counts exclude frontmatter, code blocks, display equations, and Sources; image captions are included. All articles satisfy the requested 1,800–2,500-word range.

## Integration

- The interactive roadmap includes both new series and all 171 technical article links.
- Whole-series difficulty and topic tags remain separate from article topic groups.
- Accurate new topic filters identify **Model Compression** and **Vision & Generative Models**.
- Six existing hybrid, checkpoint-configuration, architecture-comparison, quantization, multimodal, and post-training articles now link to the deeper foundations.
- Metadata is stored in the editorial content.db and exported to the published snapshot; article bodies and original SVG/PNG assets remain in the content collection.

## Verification

- Production build and published database validation passed.
- All 171 technical articles passed word-length, equation-presence, KaTeX rendering, content-path, internal-link, and image-asset checks.
- All 11 series passed consecutive display numbering and Previous/Next checks.
- Roadmap selection, goal paths, progress, persistence, and reset/cancel passed through the actual client script in a controlled DOM harness.
- All 10 topic filters, combined subject/difficulty filters, tag search, empty state, reset, and View more behavior passed through the actual catalog client script.
- All 54 distinct new-article source URLs were checked. 52 returned HTTP 200; 2 ACM DOI pages block automated access with HTTP 403. No missing-page response was found.
- Original illustrations were rendered and inspected locally; unsupported subscript-letter glyphs were replaced with readable notation.

No GPU model training, architecture search, or inference benchmark was executed. Numerical examples explain algebra and accounting rather than claim measured acceleration or quality. Browser-based visual QA was unavailable under the host policy; build, client-script, HTTP, and local-image verification provide the recorded evidence.
