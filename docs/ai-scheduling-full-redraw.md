# AI scheduling full figure redraw

Replaced all 75 figures across the 14 published AI scheduling articles. This revision supersedes the assets described in the previous figure-review report.

The updated blog-figure-maker guided 47 illustrated technical diagrams and 28 native quantitative redraws. Illustrated diagrams use recognizable GPU/server objects, rank groups, memory blocks and explicit dependencies. Quantitative figures retain exact arithmetic, axes and units in editable SVG. All examples remain illustrative and bounded by the article sources.

Every final figure was inspected. Corrections include unsupported invented scores, job and rank counts, checkpoint/recovery progress, cleanup before confirmed release, collective dependencies, and categorical prediction support. Old SVG files for bitmap replacements were removed from the published asset folders and archived with the canonical material. Prompts/specifications, hashes and findings are retained in the canonical figure manifests.

Article body prose remains unchanged. Two alt descriptions were revised to describe the new candidate-filtering and durable-progress figures. Their numbers match the existing article examples.

## Figure findings

- sched-1/section-overview.png: Revised Runtime gate to executable kernels/configuration. Routed launch-failure confirmed cleanup to returned resources, separate from successful completion. Inspected corrected final bitmap.
- sched-1/deep-dive-component-01.png: Rebuilt with resolved constraint fields and exactly R0–R7 in 2 TP4 groups. Removed extra cluster devices and ambiguous tuple symbols. Inspected final bitmap.
- sched-1/deep-dive-component-02.png: Exactly 12 devices in A/B/C/D, memory obstacle at B3, 3 candidate sets and waiting condition match article. Removed invented numeric scoring table; only supported performance, policy and provenance remain.
- sched-1/deep-dive-component-03.png: Exact A/B four-device alternatives; TP8 reservation separate from backfill; completion band and cleanup before protected start are schematic.
- sched-1/deep-dive-component-04.png: Completion/failure/stop branches converge on cleanup then confirmed release; retry is conditional and follows release.
- sched-2/section-overview.png: Reduced density and removed arbitrary configuration values. Training groups and serving demand converge through separate hard envelopes and supported estimates; policy and later observations are visible.
- sched-2/deep-dive-component-01.png: Executed 60000 vs durable 59500 produces 40000 vs 40500 remaining; 500-step nondurable gap explicit.
- sched-2/deep-dive-component-02.png: 2 TP4 groups contain R0–R3 and R4–R7. Local collective paths and cross-replica synchronization distinct; source scope retained.
- sched-2/deep-dive-component-03.png: 2 requests/s × 1500 input tokens = 3000 input tokens/s; × 200 output tokens = 400 output tokens/s. Input/output rates and SLO remain separate.
- sched-2/deep-dive-component-04.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-3/section-overview.png: All hard predicates feed feasibility; rejection separate from legal survivors, and estimate support distinct from legal-candidate ranking.
- sched-3/deep-dive-component-01.png: Per-rank memory, executable software, legal groups, ownership and coordinated launch are independent required evidence; no invented capacities.
- sched-3/deep-dive-component-02.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-3/deep-dive-component-03.png: Hardware/format/kernel/runtime/checkpoint/launch chain; missing kernel-runtime support rejects admission.
- sched-3/deep-dive-component-04.png: Categorical evidence avoids a false numeric axis; 3072 inside 2048–4096, 16384 outside with no inherited coverage.
- sched-3/deep-dive-component-05.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-4/section-overview.png: Representative forward/backward dependency paths feed collective after gradients; continuous critical path reaches update.
- sched-4/deep-dive-component-01.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-4/deep-dive-component-02.png: Trace dependency graph, operation-cost evidence and chronological residuals converge into a supported duration estimate. Independent memory/launch requirement remains explicit; qualitative plots marked illustrative.
- sched-4/deep-dive-component-03.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-4/deep-dive-component-04.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-5/section-overview.png: Snapshot and declared arrivals feed conditional remainders and same-policy replay; observed outcomes update later predictions.
- sched-5/deep-dive-component-01.png: Both eight-device domains have three occupied/five idle; aggregate ten idle cannot satisfy intact TP8.
- sched-5/deep-dive-component-02.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-5/deep-dive-component-03.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-5/deep-dive-component-04.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-5/deep-dive-component-05.png: Current snapshot, explicit arrival scenarios and changed-policy comparison have distinct conditioning labels.
- sched-6/section-overview.png: Predictor, held-out calibration, categorical supported region, abstention and deterministic hard constraints are distinct; incomplete axis removed, release label inside canvas.
- sched-6/deep-dive-component-01.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-6/deep-dive-component-02.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-6/deep-dive-component-03.png: ACI miss updates alpha 0.1 to 0.082 and cover to 0.102 with step 0.02; wider/narrower future interval direction and lack of per-action safety guarantee explicit.
- sched-6/deep-dive-component-04.png: Supported TP4 and length range separated from unsupported TP8/16384; abstention does not reject independently feasible execution or transfer coverage.
- sched-7/section-overview.png: Two local TP groups share replica synchronization path; traffic and collective cost distinguished from queued-shape fragmentation policy and revalidation.
- sched-7/deep-dive-component-01.png: Ranks R0–3 and R4–7 form two four-rank groups with local collectives; cross-replica synchronization traverses shared fabric.
- sched-7/deep-dive-component-02.png: Measurement identity includes subset, collective, illustrative 8 GiB payload and runtime/library/topology/concurrency; idle baseline distinct from shared-traffic effective cost and freshness.
- sched-7/deep-dive-component-03.png: Compact four-device occupancy preserves one intact eight-device domain; split two-plus-two occupancy leaves twelve aggregate idle but no intact TP8 domain.
- sched-8/section-overview.png: Generic GPU classes and supported feasible cells feed cost/duration/energy comparisons; no named hardware measurements or frontier values, observed outcomes inform later profiles.
- sched-8/deep-dive-component-01.png: One generic A or two generic B devices can fit fixed workload while one B cannot; profile identity includes count, layout, placement, power cap, numerical format and software/evidence.
- sched-8/deep-dive-component-02.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-8/deep-dive-component-03.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-8/deep-dive-component-04.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-8/deep-dive-component-05.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-9/section-overview.png: Corrected factorization and counted ranks: TP4×PP2×DP2 and TP8×PP2×DP1 each sixteen; both feed joint physical placement/evaluation with fixed workload intent.
- sched-9/deep-dive-component-01.png: Layout A has two replicas × two stages × four ranks; layout B one replica × two stages × eight ranks, both sixteen. Fixed intent does not imply equal memory/network or schedule.
- sched-9/deep-dive-component-02.png: Two homogeneous four-rank TP stages exchange activation across a heterogeneous pipeline boundary; shape/format/schedule checks explicit without fabricated timing.
- sched-9/deep-dive-component-03.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-9/deep-dive-component-04.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-10/section-overview.png: Shared calendar shows current gap and protected future reservation; launch/execution/cleanup/margin envelope branches into release versus timeout fallback outcomes.
- sched-10/deep-dive-component-01.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-10/deep-dive-component-02.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-10/deep-dive-component-03.png: Overrun fallback alternatives are distinct. Checkpoint durability then cleanup complete precedes all-free GPU/tensor/memory state; timeout termination does not claim instant release.
- sched-11/section-overview.png: Policy eligibility and hypothetical compact/scattered released shapes precede hard feasibility and recovery-cost selection. Rejection causes no interruption. Corrected chosen-set connector enters stop/checkpoint first, then confirmed exit/release, reservation and verified launch.
- sched-11/deep-dive-component-01.png: Priority relation alone does not override protected checkpoint phase; protected A excluded while eligible preemptible B considered, phase/owner/policy revalidation at commit.
- sched-11/deep-dive-component-02.png: Two sets each release exactly eight devices; split four-plus-four does not satisfy intact TP8 whereas whole-domain eight does; free/occupied legend explicit.
- sched-11/deep-dive-component-03.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-11/deep-dive-component-04.png: Counted sixteen GPUs in one A job versus two B jobs of eight; exposure4 versus6+6=12GPU-h; incoming request size not invented; hard protection precedes victim-count comparison.
- sched-11/deep-dive-component-05.png: Checkpoint pending branches to durable success versus timeout policy; process exit and full resource release precede reservation; partial release waits; gang and runtime failures do not reach verified allocation.
- sched-12/section-overview.png: Entitlement/current versus borrowed capacity distinguished; owner demand requires notice/checkpoint and confirmed reclaim before launch; priority ordering distinct from service accounting window.
- sched-12/deep-dive-component-01.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-12/deep-dive-component-02.png: Exact 8×8 maps: before A8+B own32+B borrowed24=64; sixteen borrowed devices change to A, after A24+B own32+B borrowed8=64. Legal shape, checkpoint and confirmed release precede launch.
- sched-12/deep-dive-component-03.png: Large training shape and urgent small evaluations explain starvation risk; strict priority compared with explicit fairness protection, accounting window and tenant-level splitting audit.
- sched-12/deep-dive-component-04.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-13/section-overview.png: Incumbent allocation and spare devices branch into supported resize, interruptible harvest or holding capacity; transitions/reclaim and shared bottlenecks shown without claimed measured gains.
- sched-13/deep-dive-component-01.png: Exactly eight current and sixteen target ranks; supported bidirectional boundary does not imply arbitrary twelve-rank support; temporary memory/state correctness and return/recovery checked.
- sched-13/deep-dive-component-02.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-13/deep-dive-component-03.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-13/deep-dive-component-04.png: Incumbent and harvester share GPU/memory/network/storage/power; retained progress/tail SLO distinguished from utilization and unsupported interference triggers stop/fallback.
- sched-13/deep-dive-component-05.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-14/section-overview.png: Common observations branch into independent policy replays and paired outcomes; shadow pass permits bounded rollout, fail/timeout fallback; corrected feedback returns to future available observations.
- sched-14/deep-dive-component-01.png: March decision inputs contain only earlier/current available evidence, with future explicitly unavailable; April late label remains outside boundary; paired policies share only observations available at each decision.
- sched-14/deep-dive-component-02.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.
- sched-14/deep-dive-component-03.png: Outcome audit distinguishes queue class/topology/tails, ownership windows, useful retained progress, occupancy without progress, and reservation/launch/release violations; conceptual curves have no benchmark values.
- sched-14/deep-dive-component-04.png: Common arrival/runtime/fault scenarios branch into independent fixed incumbent/proposed policies then join on paired difference; branch future state is not consumed by the other policy.
- sched-14/deep-dive-component-05.png: Shadow certificate mutates nothing; hard checks pass into bounded trial and observed real transition, fail/timeout into fallback with known release; stated invariants do not claim all-fault safety.
- sched-14/deep-dive-component-06.png: Reviewed rendered quantitative figure: labels and axes readable; arithmetic, units, illustration boundary and event/partition semantics preserved; final native bounds scan clear.

## Build verification

The build passed with 246 pages. All 89 optimized article and cover variants reproduce the 75 approved source figures byte for byte through the existing image service. All 14 articles pass the current humanizer measurements; their body prose and prior article-review rubric findings are preserved. Canonical and blog article copies are byte-identical. Live publication verification is recorded separately after deployment.
