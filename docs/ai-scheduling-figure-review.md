# Scheduling series figure review — 2026-09-15

Reviewed 75 figures across all 14 articles using blog-figure-maker and its figure-style reference. The previous review accepted too many readable card layouts without requiring the diagram to explain the mechanism.

Retained the first article’s 5 mechanism illustrations. Revised the other 70 editable SVG figures and exported opaque 1200 × 760 PNGs. Existing native diagrams were revised directly, consistent with the imagegen skill’s exception for code-native/vector assets.

## Findings and corrections

- 40 previous grids or tables mainly repeated labels. Replacements use actual GPU occupancy maps, communication groups, parallel branches, resource envelopes, and scaled timelines where these relationships matter. Identity and eligibility matrices remain tables where comparison is the learning objective.
- Paired policy evaluation was drawn in sequence. It now branches from common scenarios and joins on a paired difference.
- Release and timeout were drawn as a single progression. They now appear as separate success and failure outcomes.
- Fragmentation now compares equally idle device counts with different feasible 8-GPU domains.
- Rank identities, cost denominators, resize break-even points, cleanup overhead, and future-label leakage are explicit.
- All synthetic numbers remain illustrative, with source boundaries stated in each image. No new paper measurements or hardware capacity claims were introduced.

## Verification

Inspected all 13 final rendered contact sheets, covering every revised figure. Checked visible labels, arrows, numerical examples, and the fit of legends and captions. Corrected collisions during iteration. Final automated canvas bounds scan reports zero clipping candidates; this supplements visual inspection rather than replacing it. White backgrounds are opaque, so figure contrast does not depend on the page palette.

Article prose is unchanged; its previous humanizer and blog-reviewer results remain applicable. This review addresses visual teaching quality, which the previous prose and coverage checks did not establish.

## Figure inventory

- sched-2/section-overview.png: Workload contract — reviewed rendered revision.
- sched-2/deep-dive-component-01.png: Training request — reviewed rendered revision.
- sched-2/deep-dive-component-02.png: TP4 × DP2 — reviewed rendered revision.
- sched-2/deep-dive-component-03.png: Serving demand — reviewed rendered revision.
- sched-2/deep-dive-component-04.png: Per-rank memory envelope (GiB) — reviewed rendered revision.
- sched-3/section-overview.png: Separate decision boundaries — reviewed rendered revision.
- sched-3/deep-dive-component-01.png: Hard predicates — reviewed rendered revision.
- sched-3/deep-dive-component-02.png: Peak admission memory (GiB) — reviewed rendered revision.
- sched-3/deep-dive-component-03.png: Executable support chain — reviewed rendered revision.
- sched-3/deep-dive-component-04.png: Prediction support — reviewed rendered revision.
- sched-3/deep-dive-component-05.png: Recovery feasibility — reviewed rendered revision.
- sched-4/section-overview.png: Duration evidence pipeline — reviewed rendered revision.
- sched-4/deep-dive-component-01.png: Overlap changes the critical path — reviewed rendered revision.
- sched-4/deep-dive-component-02.png: Complementary evidence — reviewed rendered revision.
- sched-4/deep-dive-component-03.png: Job phase accounting (minutes) — reviewed rendered revision.
- sched-4/deep-dive-component-04.png: Layout break-even — reviewed rendered revision.
- sched-5/section-overview.png: Wait-time scenario estimator — reviewed rendered revision.
- sched-5/deep-dive-component-01.png: Snapshot objects — reviewed rendered revision.
- sched-5/deep-dive-component-02.png: Condition on survival — reviewed rendered revision.
- sched-5/deep-dive-component-03.png: First legal release wins — reviewed rendered revision.
- sched-5/deep-dive-component-04.png: Nearest-rank quantiles — reviewed rendered revision.
- sched-5/deep-dive-component-05.png: Forecast conditions — reviewed rendered revision.
- sched-6/section-overview.png: Uncertainty controls — reviewed rendered revision.
- sched-6/deep-dive-component-01.png: Different events — reviewed rendered revision.
- sched-6/deep-dive-component-02.png: CQR rank correction — reviewed rendered revision.
- sched-6/deep-dive-component-03.png: ACI feedback direction — reviewed rendered revision.
- sched-6/deep-dive-component-04.png: Support and abstention — reviewed rendered revision.
- sched-7/section-overview.png: Topology-aware decision — reviewed rendered revision.
- sched-7/deep-dive-component-01.png: Groups and physical paths — reviewed rendered revision.
- sched-7/deep-dive-component-02.png: Bandwidth measurement identity — reviewed rendered revision.
- sched-7/deep-dive-component-03.png: Fragmentation is queue-dependent — reviewed rendered revision.
- sched-8/section-overview.png: Compare configurations — reviewed rendered revision.
- sched-8/deep-dive-component-01.png: Configuration identity — reviewed rendered revision.
- sched-8/deep-dive-component-02.png: Illustrative serving mix — reviewed rendered revision.
- sched-8/deep-dive-component-03.png: Fixed training work — reviewed rendered revision.
- sched-8/deep-dive-component-04.png: Power and energy — reviewed rendered revision.
- sched-8/deep-dive-component-05.png: Pareto comparison — reviewed rendered revision.
- sched-9/section-overview.png: Joint layout and placement — reviewed rendered revision.
- sched-9/deep-dive-component-01.png: Supported layout candidates — reviewed rendered revision.
- sched-9/deep-dive-component-02.png: Heterogeneous pipeline boundary — reviewed rendered revision.
- sched-9/deep-dive-component-03.png: Stage balance — reviewed rendered revision.
- sched-9/deep-dive-component-04.png: Resize horizon — reviewed rendered revision.
- sched-10/section-overview.png: Reservation-aware backfill — reviewed rendered revision.
- sched-10/deep-dive-component-01.png: Protected resource gap — reviewed rendered revision.
- sched-10/deep-dive-component-02.png: Planning envelope in minutes — reviewed rendered revision.
- sched-10/deep-dive-component-03.png: Overrun contract — reviewed rendered revision.
- sched-11/section-overview.png: Victim selection — reviewed rendered revision.
- sched-11/deep-dive-component-01.png: Eligibility first — reviewed rendered revision.
- sched-11/deep-dive-component-02.png: Released count versus shape — reviewed rendered revision.
- sched-11/deep-dive-component-03.png: Recovery exposure — reviewed rendered revision.
- sched-11/deep-dive-component-04.png: Define smallest — reviewed rendered revision.
- sched-11/deep-dive-component-05.png: Verify the action — reviewed rendered revision.
- sched-12/section-overview.png: Ownership contract — reviewed rendered revision.
- sched-12/deep-dive-component-01.png: Illustrative 64-GPU shares — reviewed rendered revision.
- sched-12/deep-dive-component-02.png: Reclaim borrowed capacity — reviewed rendered revision.
- sched-12/deep-dive-component-03.png: Priority and fairness — reviewed rendered revision.
- sched-12/deep-dive-component-04.png: Token service cost — reviewed rendered revision.
- sched-13/section-overview.png: Spare-capacity alternatives — reviewed rendered revision.
- sched-13/deep-dive-component-01.png: Resize capability — reviewed rendered revision.
- sched-13/deep-dive-component-02.png: Permanent resize horizon — reviewed rendered revision.
- sched-13/deep-dive-component-03.png: 30-minute trough budget — reviewed rendered revision.
- sched-13/deep-dive-component-04.png: Interference constraints — reviewed rendered revision.
- sched-13/deep-dive-component-05.png: Temporary resize sequence — reviewed rendered revision.
- sched-14/section-overview.png: Decision-system evaluation — reviewed rendered revision.
- sched-14/deep-dive-component-01.png: Information boundary — reviewed rendered revision.
- sched-14/deep-dive-component-02.png: Resource-weighted error — reviewed rendered revision.
- sched-14/deep-dive-component-03.png: Outcome dimensions — reviewed rendered revision.
- sched-14/deep-dive-component-04.png: Paired experiments — reviewed rendered revision.
- sched-14/deep-dive-component-05.png: Guarded action rollout — reviewed rendered revision.
- sched-14/deep-dive-component-06.png: 80 GPU-hour occupancy ledger — reviewed rendered revision.
