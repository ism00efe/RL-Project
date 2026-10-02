# evo — Plan

## Vision
Creatures start from the same simple form and co-evolve body and movement under different environments/tasks. Distinct "species" emerge (quadruped, snake-like, biped…) — designed by environment and physics, not by hand.
Long term (Phase 4–5): species share one world (ecosystem); intervention experiments (remove a species, change climate, add a threat).

Principles: no LLMs (PPO + evolutionary algorithms) · no hand-scripted behavior or bodies · learning must be visible (before/after, generations) · training sim and rendering are separate (record, then replay in browser) · every phase is completable and showable on its own.
Out of scope for now: flight (land first; swimming optional after Phase 3); photorealism (capsule/box primitives, aesthetics from presentation).

## Building blocks
| Need | Existing | Ours |
|---|---|---|
| Physics | MuJoCo + MJX (GPU/JAX) | — |
| RL (PPO) | MuJoCo Playground / Brax PPO | rewards, observations |
| Evolution | evosax or a simple GA | outer loop, mutation ops |
| Body repr. | reference: DERL / UNIMAL | genome → MJCF compiler |
| Environments | MuJoCo heightfields | procedural terrain generator |
| Viz | Three.js | recording format + web player |

References (direction and pitfalls, not to copy): DERL (Gupta et al., Nature Comms 2021) · DARLEI (2023, single-GPU DERL) · Karl Sims, Evolving Virtual Creatures (1994).

## Architecture decisions
1. **Graph genome (from Phase 2).** Node = body part (shape, size); edge = joint (type, axis, range, recursion/symmetry). Parametric-only genomes yield variations of one species; real diversity needs graphs. Phase 1 uses a stock model.
2. **Parametric procedural environments.** Env = (terrain type, difficulty, friction, slope, obstacle density, task). Same generator later produces regions of one world (Phase 4).
3. **Phase-independent recording format.** Body geometry (part list) + per-frame pos/rot per part + metadata (genome id, generation, env, fitness, parent id). Viewer knows only this format.
4. **Lineage tracked from day one.** Parent id stored for every individual (lineage tree, Phase 4).

Known risk: MJX is fast for many copies of one model; different bodies mean recompilation. Plan: parallel envs per body, bodies processed sequentially or in small groups. Measure compile time in Phase 3; if bottleneck: smaller population, shorter training, group similar topologies.

## Phases (no phase transition without meeting the exit criterion)

### Phase 1 — One body learns to walk
Stock quadruped (Playground) on flat ground, PPO. Save policy snapshots (start/mid/end).
Exit: trained mean forward velocity ≥ 5× random policy. Output: side-by-side start/mid/end video.

| # | Step | Verify |
|---|---|---|
| 1.1 | Env setup (WSL2 if needed), deps | `jax.devices()` shows GPU |
| 1.2 | Train a stock Playground locomotion env with defaults | final mean reward, one line |
| 1.3 | Save 3 snapshots (start/mid/end) | 3 checkpoint files exist |
| 1.4 | Fixed-seed rollouts, mean forward velocity per snapshot | 3 numbers; end ≥ 5× start |
| 1.5 | Render 3 videos, combine side by side | one video file |
| 1.6 | Define recording format v1, export final policy trajectory | file + part count |

Result (2026-10-02): end mean_vx 0.928 m/s at commanded 1.0 (start −0.004, random −0.012). The ratio criterion is meaningless when the baseline is ≈0; the user accepted Phase 1 as passed. Recording format v1: `docs/RECORDING.md`.

### Phase 2 — Genome + web player
Graph genome + compiler. 3–4 hand-written genomes (quadruped, hexapod, snake, biped), each trained. Recording format + Three.js player (ground, shadows, soft light, follow cam, generation/fitness label).
Exit: all 4 compile, are stable, train, and play in browser. Output: shareable link.
"Train" = fixed-seed mean forward velocity of the end snapshot ≥ 0.3 m/s (an absolute threshold, because ratios to a ≈0 baseline are meaningless; see Phase 1).

| # | Step | Verify |
|---|---|---|
| 2.1 | Genome schema: graph nodes (part shape, size) + edges (joint type, axis, range, recursion, symmetry), JSON (de)serialization; schema needs user approval | 4 hand-written genome files round-trip unchanged |
| 2.2 | Compiler genome → MJCF (unroll recursion/symmetry, primitives only, one motor per joint) | 4 MJCFs load in MuJoCo; parts/joints count each |
| 2.3 | Stability: 10 s passive sim per body in MJX | 4 lines: no NaN, max \|qvel\| |
| 2.4 | Generic locomotion env for any compiled body (obs: joint pos/vel, torso orientation/velocity; reward: forward velocity + alive − energy; flat ground); reward terms need user approval | env jit-steps for all 4; env-steps/s |
| 2.5 | Train each body with PPO (same hyperparameters, fixed budget, `configs/`) | 4 numbers: end mean_vx (≥ 0.3) |
| 2.6 | Export each trained policy as recording v1 | 4 files + part counts |
| 2.7 | Three.js player (Vite, `viewer/`): load recording, ground, shadows, soft light, follow cam, genome/generation/fitness label | Go1 recording plays; one screenshot |
| 2.8 | Static build + shareable link (hosting target to be chosen with user) | URL plays all 4 recordings |

Result (2026-10-02): all steps done; exit criterion met on paper (every body ≥ 0.3 m/s) but the gaits are not yet credible.
End mean_vx (m/s), base reward → upright-termination variant: quadruped 4.76 → 7.63, hexapod 4.57 → 9.07, snake 1.95 → 1.64, biped 2.26 → 5.48.
Base run: quadruped/hexapod run upside down (upright 2% of steps). Variant fixes flipping (upright 100%) but speeds are implausible
(biped skates on one sliding foot). Cause: strong motors vs. negligible energy cost, plus soft-contact foot slip. Open decision before Phase 3:
reward/energy terms, motor strength, contact settings. Player: https://claude.ai/artifact/Y4zQUNqsThGYzY7zaYMcrz (private until shared).

### Phase 2b — Physics credibility (approved by user 2026-10-02 as one block: run all steps without per-step approval)
Goal: tell real strategies from simulator exploits, and make strength cost something, before evolution can exploit it.
Principles agreed with the user: odd-looking but physically valid gaits (tumbling, crawling, hopping) are legitimate results, not failures;
simulator artifacts (feet sliding while in contact, contact penetration) are not. Energy cost is the per-step precursor of the Phase 4
energy budget (same quantity: muscle mechanical work), so adding it now is in-plan. Damage/health is deferred to Phase 4 (not a reward term now).
Exit: for each of the 4 bodies, contact slip p50 < 0.3 m/s and end mean_vx changes < 20% between default and strict sim; Froude number
Fr = v²/(g·leg_length) reported (animals run at Fr ≲ 3; above that, investigate).

| # | Step | Verify |
|---|---|---|
| 2b.1 | Sim validity test: re-evaluate existing end snapshots (`runs/phase2`, `runs/phase2_upright`) under a strict sim (smaller timestep, more solver iterations, stiffer contacts) vs default; measure contact slip | table: body × {default, strict} mean_vx + slip p50 |
| 2b.2 | Muscle model: motor strength derived from the child part's cross-section (genome `strength` → dimensionless multiplier) and a force–velocity limit (torque falls linearly to 0 at max joint speed) | 4 bodies: per-joint max torque listed; zero-action check still 0 NaN |
| 2b.3 | Energy cost: penalty ∝ Σ\|τ·ω\| (mechanical power), weight in config; log cost of transport (J/(kg·m)) | CoT per body printed by eval |
| 2b.4 | Retrain 4 bodies with 2b.2+2b.3 (+ strict sim settings if 2b.1 shows they matter), upright termination OFF (flipping allowed if it is physically valid); eval, videos, recordings, update player | per body: mean_vx, upright_frac, slip p50, CoT, Fr; player link updated |

Result (2026-10-02): 2b.1–2b.4 run. Physics artifacts gone (slip p50 0, results identical under default/strict/ultra sim), but under
reward = forward velocity − energy cost every body learned to (almost) stand still. Paused: the user never set speed as a goal; forward
velocity was a benchmark default from this plan. Open question before retraining or Phase 3: what is a creature's task / fitness, and is
not moving acceptable (plants, low-energy species)? Details in CLAUDE.md Status ("Goal gap").

Idea for later (not in 2b): anisotropic friction as a genome trait ("scales") — real snakes rely on it; our ground has isotropic friction.

### Phase 3 — Evolution in niches (main result)
Outer loop: population → mutate → short PPO each → fitness → select. Same starting form, different envs: flat, stairs, rough, slope (+ optional tasks: carry, push).
Exit: best individuals of ≥2 envs topologically distinct (leg count / structure). Output: species gallery, generational change, lineage tree.

### Phase 4 — Shared world (vision)
Species in one world, regions = envs; energy, food, reproduction. Likely lower-fidelity physics mode. Not detailed before Phase 3 is done.

### Phase 5 — Intervention experiments (vision)
Remove a species / change climate / add a threat → observe ecosystem response.

## Setup
Python 3.11+, `mujoco`, `mujoco-mjx`, `playground`, `jax[cuda12]`, `brax`; `evosax` in Phase 3. Viewer: Node + Vite + Three.js (separate folder, static build).

## Risks
| Risk | Mitigation |
|---|---|
| Envs converge to similar bodies | pick very different envs (stairs vs flat) |
| Physics exploits (vibration sliding etc.) | torque limits, energy penalty, timestep checks |
| Per-body compile time | start population ≈16–32, measure |
| 8 GB VRAM | tune parallel env count |
| Scope creep (Phase 4–5 early) | exit criteria gate every phase |
