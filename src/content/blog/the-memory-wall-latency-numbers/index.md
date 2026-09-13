---
title: 'The Memory Wall: Latency Numbers Every Engineer Should Feel'
description: "Register to RAM is a 300x cliff, RAM to SSD is 1,000x more — scale it to human time and you'll never write a pointer chase the same way again."
pubDate: 'Sep 13 2026'
heroImage: './cover.png'
code: 'mem-1'
order: 4
series: 'comp-arch'
topic: 'Memory Hierarchy'
tags: [memory, latency, hardware]
---

Reading a value from a CPU register takes about 0.3 nanoseconds. Reading the same value from main memory takes about 100 nanoseconds, roughly 300 times longer, and a random read from an SSD costs another thousand times on top of that. Those three numbers explain more real-world performance mysteries than any profiler feature I know, and most engineers have never sat down and felt how big the ratios actually are.

This article is about building that feel. We'll walk the canonical latency table, rescale it to human time, count exactly what one cache miss costs in wasted arithmetic, look at why the gap exists in the first place, and end with the place the memory wall bites hardest today: generating tokens from a large language model.

## The table, and why it looks the way it does

Every working systems engineer eventually memorizes some version of this table. It descends from a slide Jeff Dean showed at Google in 2009 ("Numbers Everyone Should Know"), which Peter Norvig had published a version of earlier, and which Colin Scott later turned into an interactive chart that extrapolates the trends year by year. The rough 2020s values:

| Operation | Latency |
|---|---|
| Register access | ~0.3 ns |
| L1 cache hit | ~1 ns |
| L2 cache hit | ~4 ns |
| L3 cache hit | ~10–20 ns |
| Main memory (DRAM) | ~100 ns |
| NVMe SSD random read | ~100 µs |
| Disk seek (HDD) | ~10 ms |
| Network round trip, same continent | ~150 ms |

Two things to notice before the numbers blur together. First, each tier is not a little slower than the one above it; the steps are factors of 4 to 1,000. Second, the table spans nine orders of magnitude, from 0.3 nanoseconds to 150 milliseconds. Human intuition is terrible at nine orders of magnitude, which is why the rescaling trick below is worth doing at least once in your life.

Quick vocabulary so nothing is taken on faith. A *register* is one of a few dozen storage slots inside the CPU core itself, physically adjacent to the arithmetic units. A *cache* is a small, fast memory on the CPU die that keeps copies of recently used data; L1, L2, and L3 are successively larger and slower levels of it. *DRAM* (dynamic random-access memory) is main memory, the "16 GB of RAM" in your laptop, sitting centimeters away across a bus. An *SSD* stores bits in flash cells and is persistent; DRAM forgets everything at power-off. *Latency* is how long one access takes from request to data; it is a different quantity from *bandwidth*, which is how many bytes per second you can stream, and confusing the two is the most common memory-performance mistake there is. We'll come back to that.

## Scaled to human time

Multiply everything by a billion, so one nanosecond becomes one second. Now the table reads like this:

- **Register: 0.3 seconds.** A blink. The data is in your hand.
- **L1 cache: 1 second.** Grabbing a pen from your desk.
- **L2 cache: 4 seconds.** Reaching a book on the shelf behind you.
- **Main memory: 100 seconds.** Almost two minutes. You walk down the hall, find the room, unlock it, pull the file. Every single time.
- **NVMe SSD: about 28 hours.** The file is in another city. You drive there, sleep over, and drive back.
- **Disk seek: about 4 months.** The file is on a container ship.
- **Cross-continent network round trip: about 5 years.** You mail a letter and wait for a reply through two elections.

![The memory hierarchy scaled so one nanosecond equals one second, from a 0.3-second register blink to a five-year network round trip](./latency-ladder.png)

The step that should reorganize your programming instincts is the third one. A modern out-of-order core can start several instructions every cycle, but only when the operands are on-chip. The moment it needs a value from DRAM, your CPU stands in the hallway for two subjective minutes, and it can do that millions of times per second without any profiler line saying "waiting."

## A worked example you can do on paper

Take a 3 GHz core. One cycle is 1/3 of a nanosecond. Suppose it can complete two integer additions per cycle, which is conservative for anything shipped in the last decade. That's 6 additions per nanosecond. A single last-level cache miss to DRAM costs about 100 ns, so:

**one DRAM miss ≈ 100 ns × 6 adds/ns = 600 additions thrown away.**

Now stretch that into a real workload. You want to sum 10 million 4-byte integers, 40 MB of data, too big for any cache.

**Case 1: the integers sit in a contiguous array.** DRAM latency barely matters here, because the access pattern is predictable. The hardware prefetcher spots the sequential stride and requests cache lines before the core asks for them, so the cost is set by bandwidth, not latency. At a sustained 20 GB/s for a single core, streaming 40 MB takes 40 MB ÷ 20 GB/s = **2 milliseconds**. The arithmetic itself (10M adds at 6 per ns) would take 1.7 ms, so compute and memory are nicely overlapped.

**Case 2: the same integers live in a linked list whose nodes were allocated over time and are scattered across the heap.** Each node's address is only known after the previous node is loaded, so the prefetcher is blind and every step is a dependent DRAM access. The cost is now latency times count:

10,000,000 nodes × 100 ns = 10⁹ ns = **1 full second.**

Same data, same number of additions, same big-O complexity. The linked list is 500 times slower, purely because of *where the bytes are* and *whether the next address is predictable*. This is why "cache-friendly data layout" is not a micro-optimization; it is routinely worth more than any algorithmic constant you will ever shave.

## Why the wall exists: compute sprinted, memory walked

None of this was inevitable. In 1980 a DRAM access cost a handful of CPU cycles and the hierarchy barely mattered. Then the trajectories split. Hennessy and Patterson's textbook has the famous chart: single-core processor performance grew around 52% per year from the mid-80s to the early 2000s, while DRAM latency improved around 7% per year. Compound those for two decades and you get a gap of several hundred times; the industry saw it coming, and Wulf and McKee named it in their 1995 paper "Hitting the Memory Wall."

![Processor versus DRAM performance since 1980 on a log scale, diverging to a roughly thousandfold gap. Redrawn from Hennessy and Patterson](./memory-gap.png)

Why couldn't DRAM keep up? Because DRAM is optimized for a different objective: cost per bit. A DRAM cell is one transistor and one capacitor, packed as densely as physics allows. Reading it means selecting a row, letting thousands of tiny capacitors dump their charge onto long wires, and waiting for sense amplifiers to resolve those faint signals into digital ones. Those analog settling times are set by wire capacitance and cell physics, and they have barely moved: the core row-activation and column-access delays have hovered around 13–15 ns across DDR2, DDR3, DDR4, and DDR5. Each generation transfers data faster once a row is open, which is a bandwidth win, but first-access latency in nanoseconds has been close to flat for twenty years. David Patterson generalized the pattern in a 2004 CACM article: across memory, disk, and network alike, latency lags bandwidth, roughly quadratically.

So architects stopped waiting for DRAM and built around it. That is the entire reason the memory *hierarchy* exists: since you can't make all memory fast, you make a little memory fast and bet on locality, the empirical fact that programs reuse recently touched data (temporal locality) and touch neighbors of recently touched data (spatial locality). Caches are that bet cast in silicon, and on a modern die they take up more area than the cores do.

## Going deeper: anatomy of one miss, and how CPUs fight back

Follow one load instruction that misses everywhere. The core computes a virtual address, translates it through the TLB (a small cache of page mappings; missing *there* adds a page-table walk on top). The L1 lookup fails in a nanosecond or so, L2 in a few more, L3 in ten to twenty. The request enters the memory controller's queue, gets scheduled onto a DRAM channel, and the chip executes its little protocol: activate the row (~14 ns), issue the column read (~14 ns), burst the 64-byte cache line back, eventually precharge the row for the next access. Add queueing and the trip across the chip, and you arrive at the ~100 ns headline number. Note the useful payload: you asked for maybe 8 bytes, and the machine moved 64, because betting on spatial locality means always fetching a full line.

The core does not simply stand still for those 100 ns. Out-of-order execution keeps a window of a few hundred in-flight instructions and executes whatever is independent of the missing load. That reliably hides an L2 miss. It cannot hide DRAM: at 3 GHz, 100 ns is ~300 cycles, and with several instructions per cycle the miss punches a hole of over a thousand issue slots, far more than any realistic window can fill with independent work. So the machine layers on more tricks: prefetchers that recognize stride patterns, memory-level parallelism (a core can keep a dozen or more misses in flight at once, which is why *independent* random accesses hurt far less than a *dependent* pointer chase), and simultaneous multithreading, which fills stall holes with another thread's instructions. Every one of these is a workaround for the same underlying number. When people say modern CPU architecture is mostly about the memory system, this is what they mean.

There is an energy version of the wall too, and it decides chip architecture as much as the time version. In Mark Horowitz's much-cited ISSCC 2014 numbers, a 32-bit add costs about 0.1 picojoules while fetching 64 bits from DRAM costs on the order of a nanojoule, a ratio of several thousand. Moving data costs vastly more than computing on it, in joules as well as nanoseconds.

## The memory wall, at datacenter scale: LLM decode

Here is the modern punchline. When a large language model generates text, it produces one token at a time, and each new token's computation must read essentially every weight of the model once while performing only about two floating-point operations per weight read. That ratio, FLOPs per byte moved, is called arithmetic intensity, and at batch size 1 it sits around 1–2. A modern accelerator needs an intensity in the hundreds to keep its math units busy.

Concretely: a 70-billion-parameter model at 16-bit precision is 140 GB of weights. An NVIDIA H100 offers 3.35 TB/s of HBM bandwidth (vendor-reported, like all peak specs). The floor for one decode step is

140 GB ÷ 3.35 TB/s ≈ 42 ms per token, or about **24 tokens per second**,

no matter that the same chip advertises near a petaflop of tensor throughput. During single-stream decode the multipliers idle at under 1% utilization; the workload is a pure bandwidth play. This is exactly the linked-list lesson at warehouse scale: performance set by data movement, with compute along for the ride.

![Decode math for a 70B model: 140 GB of weights read per token over 3.35 TB/s of HBM gives a 42 ms floor, about 24 tokens per second](./decode-wall.png)

The entire modern inference stack is a response to this. Batching lets N concurrent requests share one read of the weights, multiplying arithmetic intensity by N. Quantization to 8 or 4 bits shrinks the bytes that must move. KV caches, speculative decoding, HBM stacked ever higher and wider: all of it is memory-wall engineering. It's why I keep insisting that [an ML performance engineer's job](/blog/what-does-an-ml-performance-engineer-do/) is mostly moving bytes, why [goodput and utilization tell such different stories](/blog/goodput-vs-utilization/) on decode-heavy fleets, and why the [Blackwell-to-Rubin roadmap is best read as memory math](/blog/blackwell-to-rubin-memory-math/) rather than FLOPs math. And if the pipeline mechanics of a core stalling on a load are fuzzy, the picture in [What a CPU Actually Does](/blog/what-a-cpu-actually-does/) is the prequel to this article.

## Common misconceptions

**"DDR5 is way faster than DDR4, so memory latency is improving."** Faster here means bandwidth. DDR5 moves more bytes per second through prefetching wider chunks and running the interface faster, but the time from a cold request to first data is still governed by row activation and sensing, around 14 ns internally and ~80–100 ns load-to-use, essentially unchanged since DDR2. If your workload is a dependent pointer chase, a DDR5 upgrade does approximately nothing.

**"The CPU's out-of-order magic hides memory latency, so I don't need to think about it."** It hides *some*. A few hundred instructions of lookahead cover an L2 or often an L3 miss. A DRAM miss is a 300-cycle, thousand-slot bubble; nothing in a realistic window fills it. Measure a workload that misses to DRAM every 30 instructions and you'll find the core spending well over half its time stalled, while `top` cheerfully reports 100% CPU. The workaround that actually works, memory-level parallelism, only helps when your accesses are independent, which is a property of *your data structure*, not of the hardware.

**"SSDs are so fast now that RAM barely matters."** An NVMe flash read at ~100 µs is a genuine miracle next to a 10 ms disk seek. It is also a thousand times slower than DRAM. In human scale: two minutes versus 28 hours. Any system that treats flash as "slightly slower memory" (rather than as a different tier with its own access-size and queueing rules) gets destroyed by this ratio, which is precisely why databases still obsess over buffer pools and why "it fit in RAM" remains the best performance fix in the industry.

## Takeaway

- The hierarchy steps are multiplicative cliffs: ~1 ns L1, ~100 ns DRAM, ~100 µs SSD. One DRAM miss forfeits several hundred additions; layout and access patterns routinely beat algorithmic constants.
- The wall is physics plus economics: DRAM optimizes cost per bit, so its latency has been nearly flat for decades while bandwidth (and compute) compounded. Caches, prefetchers, and out-of-order execution are all workarounds for that one flat line.
- LLM decode is the memory wall wearing a datacenter badge: ~2 FLOPs per byte means token rate is bandwidth divided by model bytes. Judge accelerators, batching schemes, and quantization through that lens first.

## Sources

- Colin Scott, "Latency Numbers Every Programmer Should Know" (interactive): https://colin-scott.github.io/personal_website/research/interactive_latency.html
- Ulrich Drepper, *What Every Programmer Should Know About Memory* (2007): https://people.freedesktop.org/~lkml/cpumemory.pdf
- W. A. Wulf and S. A. McKee, "Hitting the Memory Wall: Implications of the Obvious," *ACM SIGARCH Computer Architecture News* 23(1), 1995. DOI: 10.1145/216585.216588
- J. L. Hennessy and D. A. Patterson, *Computer Architecture: A Quantitative Approach*, 5th ed., Morgan Kaufmann, 2012 (processor–DRAM gap figure).
- D. A. Patterson, "Latency Lags Bandwidth," *Communications of the ACM* 47(10), 2004.
- M. Horowitz, "Computing's Energy Problem (and what we can do about it)," ISSCC 2014 keynote (operation energy table).

*Part of the **Computer Architecture & ASIC** series. Previous: [What a CPU Actually Does](/blog/what-a-cpu-actually-does/). Next up: caches — how a few megabytes of SRAM hide a hundred-nanosecond problem.*
