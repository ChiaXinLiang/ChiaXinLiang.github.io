---
title: 'From DRAM to HBM: How Memory Went 3D'
description: "A DDR5 module moves ~50 GB/s; an HBM3e package moves 8 TB/s. The 160× gap comes from geometry, not faster cells — here is the full story."
pubDate: 'Sep 13 2026'
heroImage: './cover.png'
code: 'mem-3'
order: 10
series: 'comp-arch'
topic: 'Memory Hierarchy'
tags: ['dram', 'hbm', 'memory']
---

A server-grade DDR5 module delivers about 50 GB/s. The HBM3e memory sitting on a flagship AI GPU delivers 8 TB/s per package, 160 times more. Both are built from the same one-transistor memory cell that IBM's Robert Dennard patented in 1968, and their wires toggle at broadly similar speeds. The entire 160× gap comes from geometry: how many wires you can attach, and how far the bits have to travel. This article walks from a single DRAM cell up to a 3D stack of silicon, and shows you the arithmetic along the way.

## One transistor, one leaky bucket

Strip away every acronym and a DRAM bit is astonishingly simple: one transistor and one capacitor. The capacitor is a tiny bucket for electric charge. Full bucket means 1, empty bucket means 0. The transistor is a switch that connects the bucket to the outside world when, and only when, its row is selected.

The wiring follows a grid. A horizontal **wordline** connects to the transistor gates of every cell in a row; raising it switches all those transistors on at once. A vertical **bitline** connects to one cell per row and carries the charge out. This grid is why DRAM addresses come in row and column halves, and why your memory chip is physically a vast checkerboard of identical cells — billions of them per die.

![Schematic of a single DRAM cell: wordline, bitline, access transistor, storage capacitor, and sense amplifier, annotated with charge and refresh numbers](./one-bit-of-dram.png)

The bucket is absurdly small. A modern cell capacitor holds roughly 10–30 femtofarads. Charge it to a fraction of a volt and you have stored a few tens of thousands of electrons. That is the physical entirety of one bit of your data: about 40,000 electrons, held in a structure etched so deep and narrow that its aspect ratio resembles a drinking straw a metre long.

Three consequences fall straight out of this design, and they shape everything above it.

First, **reads are destructive**. To read a cell, the bitline is precharged to a midpoint voltage, the wordline fires, and the capacitor's charge dribbles onto the bitline, nudging its voltage by a few tens of millivolts. A **sense amplifier** at the end of the bitline detects the nudge's direction and snaps to a full 0 or 1. In doing so the cell's charge is consumed, so the sensed value must be written back before the row closes.

Second, **the bucket leaks**. Transistors are imperfect switches, and 40,000 electrons do not stay put. Every cell must be refreshed — read and rewritten — within 64 milliseconds per the JEDEC standard, faster when hot. The D in DRAM, *dynamic*, is a polite word for "forgets constantly."

Third, **the cell itself has barely gotten faster**. The time to open a row, sense it, and restore it is set by analog physics: tiny charge, long wires, minuscule voltage swings. Row-access latency has hovered in the tens of nanoseconds since the DDR2 era. Nearly all the improvement in memory since then has gone into moving *more bits at once*, not fetching one bit sooner.

## The pin problem

If cells don't get faster, bandwidth has only two levers:

**bandwidth = number of data wires × bits per second per wire**

A standard DDR5 module exposes 64 data wires (organized as two independent 32-bit subchannels). Every generation of DDR has pushed the second lever, per-wire speed: DDR3 at 1.6 Gb/s per pin, DDR4 at 3.2, DDR5 now at 6.4 and climbing. But those bits travel roughly ten centimetres across motherboard traces, through a socket, a connector, and stub-riddled topology. At multi-gigabit rates that path behaves like a bad radio channel; it demands careful termination, equalization, and training, and each doubling costs disproportionate signal-integrity effort and energy. Sending a bit off-package over a board costs on the order of 10 picojoules or more, most of it spent just driving the wire.

The first lever, adding wires, hits a wall even sooner: pins. Every data wire needs a pin on the memory package, a trace on the board, and a pin on the processor package. A big server CPU already spends thousands of its pins on a dozen memory channels; the board around the socket is a dense forest of length-matched traces. You cannot route ten thousand data wires through a motherboard. At PCB scale, wires are a scarce resource.

So here is the trap circa 2013, when GPU designers saw compute throughput doubling on schedule while memory bandwidth crawled: cells can't clock faster, boards can't hold more wires, and per-pin speed is an expensive treadmill. The way out was to stop treating memory as a thing you plug into a board, and start treating it as a thing you build *next to the processor* — and up.

## The worked example: run the numbers yourself

Grab a napkin; the arithmetic is genuinely this short.

**DDR5-6400 module.** 64 data wires, each moving 6,400 megatransfers per second, one bit per transfer per wire:

> 64 wires × 6.4 Gb/s = 409.6 Gb/s = **51.2 GB/s per module**

A high-end server CPU with 12 channels of DDR5-6400 reaches 12 × 51.2 ≈ **614 GB/s**, and pays thousands of package pins for it.

**One HBM3e stack.** The interface is 1,024 data wires — sixteen times a DIMM — each running up to 9.6 Gb/s:

> 1,024 wires × 9.6 Gb/s = 9,830 Gb/s ≈ **1.2 TB/s per stack**

**One GPU package.** A flagship accelerator surrounds its compute die with 8 stacks. Shipping parts clock slightly below the per-pin maximum, landing at:

> 8 stacks × ~1 TB/s ≈ **8 TB/s per package**

![Bar comparison: 64 wires at 6.4 Gb/s gives 51.2 GB/s for DDR5, while 1,024 wires at 9.6 Gb/s gives 1.2 TB/s for one HBM3e stack](./width-times-rate.png)

Notice what did *not* change: per-wire speed. 6.4 versus 9.6 Gb/s is a factor of 1.5; GDDR7 graphics memory actually runs its pins three times faster than HBM3e. The 160× package-level gap is almost entirely width, with a helping of stack count. HBM is not fast memory. HBM is *wide* memory, and the whole trick is making that width physically routable.

## Going 3D: TSVs and the interposer

Width was unroutable on a motherboard, so HBM abandons the motherboard. Two structural moves make 1,024-wire interfaces possible.

**Move one: stack the dies.** An HBM package is 8 to 12 DRAM dies (16 in HBM4's tallest configurations) stacked vertically on top of a base logic die. Each DRAM die is ground down to a few tens of microns thick — thinner than a human hair — and connected to its neighbors by **through-silicon vias**, or TSVs: holes etched straight through the silicon and filled with copper, thousands of them per die, at pitches measured in tens of microns. Signals travel between dies over distances of microns rather than centimetres. The base die at the bottom of the stack collects all of it, handles test and repair logic, and presents the 1,024-bit interface to the outside.

![Cross-section of an HBM package: stacked thinned DRAM dies with vertical TSVs on a base logic die, sitting beside the GPU die on a silicon interposer over the package substrate](./inside-an-hbm-stack.png)

**Move two: shrink the board to a chip.** Even 1,024 wires from stack to processor cannot cross a normal package substrate, whose wiring pitch is too coarse. So both the GPU die and the HBM stacks are mounted on a **silicon interposer**: a slab of silicon, patterned with the same lithography used for chips, serving purely as ultra-fine wiring. Where a PCB ball-grid pitch is around 800 microns, interposer microbumps sit at roughly 50-micron pitch, a couple of hundred times more connections per unit area. The stack sits millimetres from the GPU, and 1,024 traces cross the gap without breaking a sweat. TSMC's CoWoS ("chip-on-wafer-on-substrate") is the best-known industrial version of this assembly.

The short, dense links pay a second dividend: energy. A signal crossing millimetres of interposer needs no termination resistors and a fraction of the drive strength of a board trace. HBM access energy is commonly cited around 3–4 pJ/bit against 10–15+ for board-level DDR — figures vary by generation and methodology, but the several-fold advantage is robust. When a GPU streams 8 TB/s, that difference alone is worth hundreds of watts.

## Going deeper: what the 1,024 wires actually are

It is tempting to picture an HBM stack as one enormously wide memory channel. It isn't, and the difference matters for performance work.

HBM3 organizes its 1,024 data wires as **16 independent channels of 64 bits each**, and each channel is further split into two 32-bit **pseudo-channels**. Every channel has its own command and address wires and its own slice of the stacked capacity, and each can have a different row open at a different address at the same time. A better mental model: an HBM stack is sixteen small, independent DRAM systems that happen to share an elevator. The memory controller earns its bandwidth by keeping all of them busy at once, which is why access *patterns* still matter enormously even with terabytes per second of headline bandwidth.

The per-pin speeds are modest *by design*. Early HBM ran 1 Gb/s per pin when DDR4 ran 2.4 and GDDR5 ran 7. Slow, short, unterminated, massively parallel links are the low-energy corner of the design space; fast narrow links are the low-pin-count corner. HBM and GDDR are the two corners, built from the same cells.

Stacking also concentrates the technology's oldest enemy: heat. DRAM retention worsens as temperature rises — above 85 °C the standard refresh interval halves — and an HBM stack sits millimetres from a die dissipating upward of a kilowatt. This is one reason the memory sits *beside* the processor rather than on top of it, and why cooling design and refresh management are quietly part of every HBM deployment.

The width lever keeps moving. HBM4 doubles the interface to **2,048 wires per stack**; SK hynix announced completed development in 2025 with mass production readiness, claiming over 40% better power efficiency than its predecessor (a vendor figure, not yet independently verified). Doubled width at similar pin speeds is how next-generation GPUs are slated to jump from 8 toward 20+ TB/s.

## Common misconceptions

**"HBM is fast memory — the cells or clocks must be quicker."** Backwards on both counts. The cells are the same 1T1C design as your laptop's DRAM, and per-pin signaling is *slower* than GDDR7 by roughly 3× and only modestly faster than DDR5. HBM's advantage is 1,024–2,048 wires against a DIMM's 64. If you remember one thing: width, not speed.

**"Stacking must have slashed latency."** It didn't, meaningfully. A random access into HBM still costs on the order of 100 nanoseconds under load, in the same neighborhood as DDR, because the dominant costs — row activation, sensing, charge restore — happen inside the same slow cell arrays. HBM attacks bandwidth, not latency. Latency is the cache hierarchy's job, which is exactly why GPUs pair enormous HBM bandwidth with large on-die SRAM and deep latency-hiding parallelism.

**"Why bother stacking? Just add more DDR channels."** Run the numbers: matching one package's 8 TB/s with DDR5-6400 modules would take about 160 channels — over 10,000 data pins plus address and control, on a socket and board that struggle past a dozen channels. The pin and routing budget of a motherboard makes the "just add channels" path physically impossible; that impossibility is the entire reason HBM exists.

## Why this is suddenly everyone's problem

For twenty years the DRAM industry optimized for cost per bit, and memory was the boring commodity under the heatsink. Large-model AI inverted that. Generating tokens from a large language model is bandwidth-bound — the whole model streams through the compute units for every token — so the memory package, not the logic die, now sets the speed limit. I ran that arithmetic in [Blackwell to Rubin: capacity stays flat, bandwidth nearly triples](/blog/blackwell-to-rubin-memory-math/), and the roadmap conclusion is blunt: GPU generations are now paced by HBM generations.

The industrial consequence is that HBM has become the choke point of the entire AI build-out. Only three companies — SK hynix, Samsung, and Micron — can make it; their HBM lines have been reported sold out more than a year ahead; and advanced packaging capacity (CoWoS and its cousins) gates how many accelerators ship regardless of how many GPU dies exist. Reports put HBM at roughly half the bill of materials of a flagship accelerator — treat the exact fraction as industry chatter, but the direction is not in dispute. The commodity became the crown jewel.

For a performance engineer, this article is the floor under two earlier ones. When [a CPU's pipeline](/blog/what-a-cpu-actually-does/) stalls for hundreds of cycles on a miss, this page is what it's waiting for: a wordline, a sense amplifier, and a trip across too few wires. And when a training cluster burns FLOPs waiting on memory, the gap shows up precisely as the difference between [goodput and utilization](/blog/goodput-vs-utilization/) — the hardware is busy, the bytes just aren't there yet.

## Takeaway

- A DRAM bit is one transistor and one leaky capacitor holding ~40,000 electrons; it must be refreshed every 64 ms, reads destroy it, and its core latency has been stuck in the tens of nanoseconds for two decades. All progress since has come from moving more bits in parallel.
- Bandwidth is wires × speed, and boards ran out of wires: DDR5's 64 pins × 6.4 Gb/s gives 51.2 GB/s per module, while HBM3e's 1,024 pins × ~9.6 Gb/s gives ~1.2 TB/s per stack — width, not clock speed, is the entire win.
- HBM gets that width by going 3D — thinned dies, copper TSVs, and a silicon interposer with 50 µm bumps — which is also why only three suppliers and a handful of packaging lines now pace the whole AI industry.

## Sources

- Ulrich Drepper, *What Every Programmer Should Know About Memory* (2007) — [people.freedesktop.org/~lkml/cpumemory.pdf](https://people.freedesktop.org/~lkml/cpumemory.pdf)
- SK hynix newsroom, *SK hynix Completes World's First HBM4 Development and Readies Mass Production* (2025) — [news.skhynix.com](https://news.skhynix.com/en/sk-hynix-completes-worlds-first-hbm4-development-and-readies-mass-production/)
- JEDEC, *High Bandwidth Memory (HBM3) DRAM*, standard JESD238 — [jedec.org](https://www.jedec.org)
- Onur Mutlu et al., memory systems lectures and DRAM research, SAFARI group, ETH Zürich — [safari.ethz.ch](https://safari.ethz.ch/)
- R. H. Dennard, "Field-Effect Transistor Memory," U.S. Patent 3,387,286 (filed 1967, granted 1968)
- Colin Scott, *Interactive Latency Numbers Every Programmer Should Know* — [colin-scott.github.io](https://colin-scott.github.io/personal_website/research/interactive_latency.html)

*Part of the **Computer Architecture & ASIC** series. Previous: the cache hierarchy that hides this page's latency, and [What a CPU Actually Does](/blog/what-a-cpu-actually-does/) for the pipeline that stalls on it. Next: what all this memory feeds — custom silicon.*
