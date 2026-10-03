# evo — Plan (v2, rewritten 2026-10-03 from the final goal; v1 is in git history, commit b209d09)

## Final goal
An evolving ecosystem. A world of adjacent biomes (e.g. rainforest, forest, steppe, desert; water open) joined by transition zones.
Simple creatures start in different biomes, evolve separately, then meet and interact (eat each other, compete, coexist).
Finally: intervention experiments (remove a species, change climate, add a threat) and watch the ecosystem respond.

## How the world works (principles)
1. **The world selects, nobody else.** A creature gains energy from food, spends it on living and moving, dies at zero energy,
   reproduces when it has enough. There is no external fitness, score or goal (no "be fast", no "reach X").
2. **Variation is random.** Offspring = parent's genome with random mutations (body, senses, brain). No creature chooses its changes;
   a change that helps survives, one that doesn't dies out.
3. **Brains are inherited.** The brain is part of the genome (neuroevolution). Optional lifetime learning on top (e.g. PPO fine-tuning,
   Lamarckian or Baldwinian) only if it wins an equal-compute comparison (stage E1).
4. **Any strategy is legitimate if physically valid:** standing still (plant-like, sloth-like), crawling, tumbling, hunting.
   Speed, CoT etc. are observations, never goals.
5. **Physics must be honest.** Strict sim, muscle torque from cross-section, energy = muscle work (from Phase 2b). Simulator exploits
   (sliding feet, flying) are bugs.
6. **Complexity grows step by step,** but every stage adds a piece of the final world; no disconnected side tasks.
7. No LLMs in the system; no hand-scripted behavior or bodies; learning must be visible (generations, lineages); sim and rendering separate.

## Foundations (done 2026-10-02) — feasibility checks, not goals
- F1 (old Phase 1): PPO pipeline on a stock quadruped; snapshots, videos, recording format v1 (`docs/RECORDING.md`).
- F2 (old Phase 2): graph genome v1 (`docs/GENOME.md`) → MJCF compiler; 4 hand-written bodies move; Three.js player (claude.ai artifact).
- F3 (old Phase 2b): physics credibility. Default sim was exploited (speeds fell up to 77% under strict sim). Now: strict sim
  (dt 1 ms, 20 iters, stiff contacts), muscle model, energy cost; slip ≈ 0 and results identical across sim settings.
  Under the benchmark reward (forward speed − energy) bodies chose to stand still: an expected, valid outcome, not a failure.
- Reusable: genome + compiler, MJX env, strict-sim settings, muscle/energy model, sim checks, recording + player. PPO stays available.

## Key technical questions (answered by measurement in E1)
| # | Question | Candidate |
|---|---|---|
| Q1 | Run many *different* bodies in one GPU batch | fixed-capacity body template (max N parts, unused parts disabled) |
| Q2 | Offspring keeps parent's brain when the body changes | body-agnostic brain: one small shared network per limb/joint + a global part |
| Q3 | Many creatures in one shared scene (interaction) | one MuJoCo scene with K creatures; many scenes in parallel |
| Q4 | Inherited-only brain vs + lifetime learning | same wall-clock budget, compare what each achieves |
| Q5 | Is the 8 GB GPU enough for sustainable evolution | creature-seconds simulated per second, at strict physics |

## Stages
Each stage: what it adds to the final world → exit criterion. Step tables are written one stage at a time and approved by the user.

### E1 — Evolution engine (feasibility of the machine, not of the creature)
Adds: the machinery every later stage runs on. Answers Q1–Q5.
Exit: mixed bodies run in one batch; a mutated child inherits and uses the parent's brain; Q4 decided by an equal-time comparison;
measured throughput → realistic population size and generations/day.

Step table (draft 2026-10-03, awaiting user approval). Order = riskiest machinery first (E1.1–E1.4 decide whether the design is
feasible at all; if E1.2 fails, stop and re-plan with the user before E1.5). All runs strict sim (F3), fixed seeds, `configs/e1_*.yaml`.

| # | Step | Verify |
|---|---|---|
| E1.1 | Fixed-capacity body template (Q1): one MJCF skeleton = root + L limb slots × S segments (proposal L=6, S=5 → 31+1 = 32 parts = current MAX_PARTS), ≤ 2 hinges per segment; genome → per-slot parameters (shape, size, attach point, dir, axes, range, strength, enabled). Disabled part = no collision, negligible mass, joint locked, motor off. Genome structure change → user approval | 4 example bodies expressed in the template: parts/joints/mass/max torques equal to `builder.compile` within 1%; zero-action 10 s 0 NaN |
| E1.2 | Mixed bodies in one batch (Q1): per-env model fields (geom size/pos, body pos/quat/mass/inertia, joint axis/range, gear, masks) batched with vmap over one template model | 4 bodies × 1024 envs in one batch: 0 NaN; first 0.2 s trajectory vs separately compiled bodies max \|Δqpos\| < 1e-3; env-steps/s mixed vs single-body |
| E1.3 | K creatures in one scene (Q3): template with K creature slots, creature–creature contacts on | K ∈ {1, 4, 16}: 0 NaN over 10 s; creature-steps/s per K |
| E1.4 | Throughput table (Q5): creature-seconds simulated per wall-second for (envs, K) grid at strict physics, GPU memory | one table; → max population × lifetime per hour |
| E1.5 | Body-agnostic brain (Q2): one small shared network per joint (inputs: own joint state, segment contact/orientation, slot id, messages from parent/children) + a global part (root state); parameter count independent of body | same parameters drive all 4 bodies in one batch; parameter count (one number) |
| E1.6 | Random mutation operators on genome v2 (body + brain): parameter noise, add/remove segment, add/remove limb, toggle hinge, brain weight noise; rates in config | 1000 mutants per example: % valid (compiles, 0 NaN 2 s); histogram of part counts |
| E1.7 | Minimal test world for E1 measurements (see open decision A below) | random brains: mean ± sd of the world's own measure over 1024 creatures (baseline) |
| E1.8 | Neuroevolution loop (`evosax`, new dependency) over mixed fixed bodies in the test world; only the world's measure ranks, no extra reward terms | measure per generation rises above random baseline; generations/hour |
| E1.9 | Inheritance (Q2 exit): evolved parent brain on its mutated children vs a random brain on the same children | children keep ≥ 50% of parent's gain over random (one ratio) |
| E1.10 | Q4: inherited-only (ES) vs ES + lifetime PPO fine-tuning (Baldwinian: learned weights not inherited), same wall-clock budget | two numbers (final measure each) + decision recorded |

Decision A (user, 2026-10-03): E1 test world = proto-E2 food arena: flat ground, scattered food pellets eaten on touch,
energy = food − basal cost − muscle work, measure = energy at end of a fixed lifetime (no reproduction yet). Needs touch +
food-direction senses already in E1. Correction: the user never ruled out goals in general; what was rejected is speed as an
imposed goal. Proposed new wording of principle 1 (awaiting user OK): "The world is the main selector (energy, death,
reproduction). No imposed tasks like speed; any other goal is added only by user decision."
Working mode proposed for E1 (awaiting user OK): one approval for the whole table + template 6×5 + evosax + brain design of E1.5;
then run steps without per-step approval, stopping only on a real blocker (E1.2 mixed-body batching fails → user picks: topology
buckets vs MJX pure-JAX impl; dependency conflict; rule 6 cases).
Genome v2 extensibility: every template part carries an open `traits` dict (defaults when absent), so new per-part traits
(eyes/vision sensor, scales = anisotropic friction, armor) are added later without breaking old genomes. Changing template
capacity (L×S) later = recompile + convert populations (lossless only when growing).
`evosax` 0.3.1 metadata: jax>=0.5, flax>=0.10 → compatible on paper with our pins (jax 0.9.2, flax 0.12.6); verify with `pip check` in E1.8.

### E2 — One biome, one lineage, a life cycle
Adds: food, energy budget (basal cost ∝ mass + muscle work), death, asexual reproduction with mutation, senses (at least touch +
a food sense) as genome traits.
Exit: the population sustains itself for many generations without extinction or explosion; bodies/brains change over generations
(lineage tree, gallery over time). What they become is not prescribed.

### E3 — Two biomes, separate evolution
Adds: biome = terrain + climate-like parameters (food density/distribution, ground friction/roughness, slope, obstacles).
Two contrasting biomes, populations isolated, same starting form.
Exit: lineages diverge in body/sense/brain traits between biomes, and the divergence repeats with a different seed.

### E4 — Connected world: transition zones and migration
Adds: biomes side by side with a transition zone; creatures can cross.
Exit: observed migration and competition at the boundary (who stays, who spreads, who disappears).

### E5 — Interaction between creatures
Adds: eating other creatures (energy transfer on contact/bite), defense traits (armor, size), collision between creatures.
Exit: at least one stable interaction pattern emerges (predator–prey, competition, coexistence) and is observable.

### E6 — Full world
Adds: more biomes (rainforest, forest, steppe, desert; water if its physics is feasible), richer traits (vision, scales =
anisotropic friction, …), possibly plant-like stationary producers.
Exit: a running multi-biome ecosystem with several coexisting species.

### E7 — Intervention experiments
Remove a species / change climate / add a threat → measure the ecosystem's response; compare against unperturbed runs.

## Decisions (user, 2026-10-03)
- Reproduction: asexual (mutated copy). Sexual reproduction maybe later.
- Water: not now (keep it simple). Adding it later is feasible if biome physics stays a per-region parameter set
  (water = region with buoyancy + drag forces applied by our code); keep that interface general.
- Compute compromise: prefer diversity over numbers. If hardware is short, cut creature count first, then some biomes.
- Plants: count as organisms (evolving if feasible; not evolving is acceptable). No sun-based plants: if everything could feed on
  sunlight, sunless biomes would be needed.

## Open questions (decide with the user when the stage comes)
- Energy input to the world: without sun, where does food come from? (e.g. biome-defined regrowth of food / plant growth rate
  as a climate parameter). Needed by E2.
- Damage/health model (needed by E5); user has no preference yet.
- Physics fidelity vs world size: exact compromise once E1 measures Q5.

## Setup
Python 3.11+, `mujoco`, `mujoco-mjx`, `playground`, `jax[cuda12]`, `brax`; `evosax` for neuroevolution (new dependency, E1).
Viewer: Node + Vite + Three.js (`viewer/`, static build). Hardware: RTX 4060 8 GB, 32 GB RAM, WSL2.

## Risks
| Risk | Mitigation |
|---|---|
| Compute not enough for a shared world at strict physics | measure in E1 before building E2; scale world/population to the measurement |
| Population dies out before evolution starts (random first brains) | generous food early; seed with briefly pre-trained brains if needed |
| Selection too noisy (luck dominates survival) | larger populations, several parallel worlds |
| One trait always wins (e.g. bigger is always better) → no diversity | square-cube physics, basal cost ∝ mass, contrasting biomes |
| Physics exploits | keep F3 sim checks in every stage |
| Drifting into side tasks | every stage must add a piece of the final world |
