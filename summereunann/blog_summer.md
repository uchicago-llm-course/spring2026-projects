# Week 1 Blog

## Insights from the Week

The question we started with was whether multi-agent protocols help on verifier-backed scientific optimization once compared against strong single-agent controls under the same visible budget. 

That question was too broad. The current data show that agent count does not predict gains. BoN, a single-model sampling baseline, beats most multi-agent systems. MoA, the one positive multi-agent protocol, gains because it synthesizes outputs from three different model families—not because it uses more agents. The operative variable is independent information coverage, moderated by what we tentatively call task composability—though that second claim rests on two tasks (Difference Bases positive, Circle Packing negative) and was not pre-specified. The rest of this note plans around that sharper framing.

Each protocol-task cell runs across seeds 1–5 under matched visible budget. Protocols may query a public practice grader during a run, but final comparison uses only the hidden final grader. The main reported quantity is MEG, which measures whether a protocol beats the strongest single-agent control at the same budget on the hidden score.

The task suite splits into four roles. MaxCut served as a diagnostic check to show that visible-grader rankings diverge from hidden-grader rankings. Difference Bases is the clean discriminative task where protocol choice creates real spread. Circle Packing is the best structured negative case—decomposition and critique do not transfer there. TSP-100, Lennard-Jones n=41, and Flat Polynomials are capability-floor exhibits. They are able to tell us where current systems hit hard limits, but not which protocol is better. Erdos overlap is pending a clean rerun but preliminary data suggest it may be the most discriminative task in the suite. The protocol scores range from 0.0 to 0.942, and Homo-chain reaches MEG = +0.105.

## Challenges and Roadblocks

The benchmark has 10 protocols, 7 tasks, and 5 seeds per cell. That is 350 cells in the primary sweep alone, plus additional cells for the brain-diversity ablation. This is manageable for a single benchmark paper, but the combinatorial explosion of protocols × tasks × seeds × backbone assignments is the binding constraint on what we can ablate. The mechanism-isolation experiment (testing whether MoA's gains come from model diversity or from the synthesis step) is relatively low-cost and is high-value, but every new ablation axis multiplies the cell count. I have had to make explicit triage decisions about which sweeps to run before submission and which to defer.

Each task uses a public practice grader and a hidden final grader. The gap between them is supposed to test generalization, but the hidden grader is something I designed. If the gap is too arbitrary, the benchmark is just measuring whether protocols happen to match my choice of perturbation. Some splits have clear structural justifications (e.g., MaxCut scores on 80% of edges publicly and 100% privately, so a protocol that reasons about graph structure should generalize). Others are less clean (e.g., Circle Packing's precision split (1e-6 vs. 1e-12) may penalize numerically aggressive protocols for reasons unrelated to solution quality). 

## Progress

Agent count is not the operative variable. MoA is positive at +0.021. BoN, which samples one model eight times, beats every multi-agent system except MoA. If adding agents were the mechanism, BoN should lose to protocols that use more of them, but it does not.

Hidden evaluation is very important for real measurement. On MaxCut, the protocol that ranks first on the visible practice grader does not rank first on the hidden final grader. Rank correlation between visible and hidden rankings is ρ = 0.828, and the top-1 slot flips. This is why the benchmark needs both graders. Without the hidden grader, MaxCut would endorse the wrong protocol.

![Practice grader rank vs. final grader rank on MaxCut. VGS ranks 1st on the practice grader but drops to 2nd on the final grader. Cross-chain ranks 7th on practice but 1st on final. The bottom half of the ranking is stable; inversions concentrate at the top.](fig1_maxcut_rank_scatter.svg)

The structure of each task determines the value of the protocol. Difference Bases is the clean positive case. For Difference Bases, MoA reaches approximately +0.179 there, and Debate also shows useful signal. Circle Packing is the clean negative case. For this case, decomposition and critique do not achieve anything. The harder numerical tasks (TSP-100, LJ n=41, Flat Polynomials) are capability floors where all protocols land at Q ≈ 0. They should not drive the MEG aggregation. Of the current seven tasks, only Difference Bases and (after the rerun) Erdos overlap produce real discriminative signal for MEG. MaxCut is a Goodhart canary excluded from aggregation, and the rest are floor or ceiling exhibits.

## Plans and Next Steps

I need to add at least one science-facing task. "Scientific optimization" in the title is vulnerable if every task is combinatorial math. Molecule QED (Card 12) is seeded and designed. Running the calibration pilot and a 17-protocol sweep on it would address the strongest content-level reviewer objection.

I will re-run Erdos after the verifier fix. This will restore the best discriminative task and fill the HPE gap.

Mixed variants, OSS variants, ToT, Single-shot, and Homo-chain do not need to appear in the first 2–3 pages once the mechanism story is locked. Cross-chain and MAgICoRe can stay as secondary checks or move to an appendix.

What carries over from the original plan are the hidden-vs-visible evaluation split, the strong single-agent control set, the task-role taxonomy (canary, discriminative, negative, floor), and the follow-up mechanism tests that ask why MoA helps when it does.
