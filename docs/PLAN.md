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
