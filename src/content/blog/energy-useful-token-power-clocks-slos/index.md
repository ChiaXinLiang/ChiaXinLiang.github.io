---
title: "Energy per Useful Token: Power Caps, Clocks, Thermals, and SLOs"
description: "Integrate measured power over useful serving work, distinguish device from system energy, and evaluate power controls under latency and capacity constraints."
pubDate: "Sep 13 2026"
updatedDate: "Sep 13 2026"
heroImage: './section-overview.png'
series: "llm-serving"
code: "energy-1"
order: 26
topic: "Cost and Energy"
level: "advanced"
tags: ["llm-serving", "ai-infrastructure"]
---

## Overview

![Concept overview: Energy per Useful Token: Power Caps, Clocks, Thermals, and SLOs. Illustrated GPU server has a power meter, thermal gauge, and clock setting.](./section-overview.png)

Reducing a GPU's reported power does not necessarily reduce the energy required to serve 1 request: if the computation takes longer the device draws power for more time, and the host and network stay active throughout. If a slower operating point causes clients to cancel or retry, the service can consume more energy per useful answer despite a lower instantaneous wattage.

Energy engineering therefore needs 2 things: a denominator tied to useful work, and a measurement boundary tied to the power being integrated. The relevant question is how many accepted outputs the system produces for its energy expenditure while satisfying latency and capacity requirements.

We will do 3 things: derive the basic accounting, examine power caps and clocks, and build an experiment that distinguishes an efficient operating point from a merely low-power one. All numerical examples are illustrative calculations, and available telemetry and control behavior vary by GPU, driver, platform, and deployment permissions.

## Deep dive

### 1. Define energy and useful work on the same interval

![Deep-dive illustration: Define energy and useful work on the same interval](./deep-dive.png)

Power is an instantaneous rate of energy consumption. For an observation interval from time a to time b, energy is the integral of measured power. If N_useful is the number of qualifying output tokens delivered during that interval, define

$$
E=\int_a^b P(t)\,dt,\qquad e_{\mathrm{token}}=E/N_{\mathrm{useful}}.
$$

Power in watts and time in seconds produce joules, but joules per token means something only once you define the token population, and the 2 obvious counters both mislead: a raw engine-generated counter can include responses discarded after cancellation, while a client-delivered counter can include incomplete objects the application rejects.

Choose 1 useful-output policy suited to the service: a conversational service may count delivered tokens in successfully completed responses, a structured-output service may count accepted complete objects instead, and an evaluation workload can report both generated work and accepted work to make the difference visible.

If no useful output completes in the interval, energy per useful token is undefined rather than 0, so report the energy and the absence of useful completions separately. This case is important during startup, failure, or overload, when considerable power consumption can coexist with little usable progress.

### 2. State whether the boundary is a device or a system

GPU telemetry measures a device-level quantity whose exact scope depends on the sensor and platform, and it does not generally represent the electricity drawn by the 6 other consumers around it: CPUs, memory, storage, networking, fans, and facility infrastructure. A device-only result should be labeled accordingly.

For several devices and separately measured host components, a simplified system boundary is

$$
E_{\mathrm{system}}=\int_a^b\left(\sum_i P_{\mathrm{GPU},i}(t)+P_{\mathrm{host}}(t)+P_{\mathrm{other}}(t)\right)dt.
$$

Avoid adding overlapping measurements: a rack power meter may already include the GPUs and host, so summing its reading with device power counts 1 part of the system twice, while adding only GPU sensors omits the host entirely. Draw the accounting boundary before collecting measurements.

Specify whether the denominator is aggregate useful output across all devices or 1 request's output. A continuously batched engine shares execution among requests, which makes exact per-request energy attribution difficult, and aggregate workload energy is usually easier to measure defensibly than assigning each concurrent request a fraction of every device sample.

Wall-level energy includes conversion losses and additional components that software telemetry may not see, and either of the 2 boundaries can support useful comparisons as long as it stays consistent across alternatives, so do not compare a device-only baseline with a wall-measured candidate and then attribute the whole difference to an algorithm.

### 3. Integrate telemetry rather than averaging mismatched counters

![Deep dive: 3. Integrate telemetry rather than averaging mismatched counters](./deep-dive-component-02.png)

Suppose samples provide power P_i at times t_i. A trapezoidal estimate is

$$
\widehat E=\sum_i\frac{P_i+P_{i+1}}{2}(t_{i+1}-t_i).
$$

The estimate requires 2 things, correctly ordered timestamps and a sampling rate adequate for the workload, and even then a short kernel can finish between coarse samples while a sensor may report a time-averaged quantity, so increasing polling frequency cannot necessarily reveal changes faster than the underlying sensor updates.

Where a supported cumulative energy counter is available, its difference over a defined interval can provide another measurement path. Verify all 4 of units, supported hardware, counter reset behavior, and documented meaning. NVIDIA's Management Library documentation distinguishes several power-related fields, and identical-looking chart labels should not be assumed to represent the same quantity.

Align the energy interval with the output-count interval: if energy includes warmup but useful tokens exclude it, the result deliberately includes startup overhead, and if you want steady-state efficiency, remove both startup energy and startup work consistently. Report the 2 cases, cold-start and steady-state, separately when both matter.

Repeat measurements and inspect variation, and preserve raw timestamps and samples so another reviewer can reconstruct the integral and confirm the selected measurement window, because 4 things can move the result: power sampling, thermal conditions, batch composition, and background activity. An apparent improvement smaller than the measurement variation should not be presented as a reliable ranking.

### 4. Derive why a lower power cap can lose efficiency

![Deep dive: 4. Derive why a lower power cap can lose efficiency](./deep-dive-component-01.png)

For approximately constant average power P and useful throughput G over a steady interval, the energy per useful token simplifies to

$$
e_{\mathrm{token}}\approx P/G.
$$

Consider an illustrative baseline using 600 W and producing 100 useful tokens per second. Its device energy is 6 J per token. A capped alternative using 450 W and producing 90 useful tokens per second reaches 5 J per token. If throughput instead falls to 60 tokens per second, the result becomes 7.5 J per token.

The power decrease is the same in both capped examples, but the throughput response determines the energy outcome. Including a constant 200 W host contribution changes the respective system values to 8, about 7.22, and about 10.83 J per token. Slower execution prolongs host energy expenditure as well.

That 600 W against 450 W comparison assumes stable useful throughput and average power, and during queue growth or changing concurrency a simple ratio can conceal an infeasible operating point. Verify that arrivals, admitted work, and useful completions are consistent with the capacity requirement before ranking the alternatives.

### 5. Connect clock sensitivity to the actual bottleneck

A workload's response to power and clock controls depends on its limiting resources, so a compute-heavy prefill can respond differently from cache-intensive decode. Memory bandwidth, matrix arithmetic, launch overhead, and CPU scheduling are 4 separate resources, and they do not all scale with the same device clock.

A simple bottleneck model is

$$
t\gtrsim\max(F/C_{\mathrm{eff}},D/B_{\mathrm{eff}})+t_{\mathrm{exposed\ overhead}},
$$

The 4 symbols are F for arithmetic work, D for relevant transferred bytes, C_eff for achieved compute rate, and B_eff for achieved bandwidth. The model omits detailed pipeline interactions but helps identify which measured quantity should move when a control changes.

If decode is primarily limited by memory movement, reducing compute capability may initially have little effect on useful throughput. However, shared resources and voltage-frequency policies can couple effects, so that outcome is a hypothesis to test. A device's power cap does not guarantee a fixed independent change in 1 performance resource.

Inspect both phase timing and achieved throughput during a sweep. If prefill slows significantly while decode remains stable, a mixed workload's first-token objective may become the binding constraint. The energy-efficient point for 1 phase is not automatically the efficient point for a service processing both.

### 6. Thermal state is part of the experiment

A short benchmark can begin on a cool device and complete before reaching the temperature and clock behavior seen in production. A long sustained workload can encounter different conditions in 3 places: cooling, fan behavior, and thermal limits. Compare candidates after a documented stabilization period.

Record observed temperatures, clocks, power, and supported throttle or event indicators. A configured clock target is different from the clock actually maintained. If 1 candidate encounters thermal limiting and another does not, the result includes that system response rather than only the nominal configuration.

Keep ambient and neighboring-load conditions as consistent as practical. Shared chassis cooling and dense accelerator placement can make 1 device's thermal behavior depend on its neighbors. A single isolated accelerator experiment does not establish the operating point for a fully populated server.

Do not disable protective behavior to obtain an attractive benchmark. The intended deployment's normal supported controls provide the relevant feasibility boundary. An efficient configuration must remain stable under the duration and environmental conditions expected by the service.

### 7. Compare energy under latency and capacity constraints

An operating point is feasible only if it meets the required service outcomes. Let the 2 constraints be G_min for required useful throughput and L_max for the applicable latency objective. A simplified selection problem is

$$
\min_x e_{\mathrm{useful}}(x)\quad\text{subject to}\quad G(x)\ge G_{\min},\quad L(x)\le L_{\max}.
$$

The control vector x may include power cap, batching policy, concurrency, and model execution configuration. Specify which latency statistic is constrained, such as p99 first-token latency for a request class. A mean latency constraint does not establish acceptable streaming gaps or tail behavior.

Batching changes 2 things at once: device efficiency improves because work per execution interval rises, and waiting time grows. An energy measurement at a large batch is useful only if the service can operate at that batch while satisfying its objective, so include scheduler waiting time, not only active GPU duration.

Plot or tabulate the feasible alternatives together. Some configurations consume less energy but cannot supply the demand; others have more headroom at a higher energy cost. A frontier of feasible choices is more informative than declaring 1 globally optimal power setting without workload context.

### 8. Include idle capacity and deployment utilization

A benchmark with continuously busy devices can omit much of a real deployment's energy use. Replicas kept warm for redundancy or bursts draw power while handling little useful traffic. Their energy belongs in a deployment-level calculation even if a saturated-kernel benchmark excludes it.

Let f be the fraction of an interval spent in a simplified active state. With active power P_active and idle power P_idle, average device power is approximately f times P_active plus 1 minus f times P_idle. Useful throughput also depends on when and how efficiently active work runs.

Reducing replica count can improve utilization and aggregate energy efficiency while reducing failure headroom or increasing queueing. Those 2 policies, autoscaling and admission, interact directly with device tuning. Evaluate them at the service level rather than treating every unused accelerator as avoidable without considering the capacity requirement.

For variable demand, integrate over representative periods instead of extrapolating from a single saturated minute. Report all 3 of the demand distribution, the scaling policy, and the warm-capacity assumptions. The same model and kernel can have very different deployment energy per useful response under different traffic patterns.

### 9. Build a sweep that produces a defensible decision

Hold 5 things the same for each candidate: model, prompt distribution, output policy, hardware boundary, and measurement method. Record the applied controls and observed operating state, then warm up execution, stabilize thermal behavior, and measure long enough to include representative batching and response completions.

For each point, report 8 quantities: useful output, energy, throughput, first-token latency, streaming gaps, completion outcomes, observed power, and observed clocks. Repeat selected points to assess variation, especially near the feasibility boundary, and mark points that violate objectives rather than letting them win the energy ranking silently.

Use profiles only after the measurements reveal a question, such as why 1 cap changes prefill more than decode. The profile can identify resource sensitivity, while the service experiment establishes the net result. Keep mathematical estimates separate from observed numbers in the report.

## Conclusion

Energy per useful output joins hardware behavior to the service's purpose. Power caps, clocks, and cooling are the 3 controls; accepted work and latency are the 2 outcomes. A sound choice reduces the integrated energy required for those outcomes at a feasible operating point, with a consistent measurement boundary and enough evidence to reproduce the comparison.

### Sources

- [NVIDIA System Management Interface documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
- [NVIDIA Management Library API reference](https://docs.nvidia.com/deploy/nvml-api/).
- [NVIDIA DCGM documentation](https://docs.nvidia.com/datacenter/dcgm/latest/).
