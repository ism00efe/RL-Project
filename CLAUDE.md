# evo — agent brief

Co-evolution of body + controller for 3D creatures (MuJoCo/MJX, PPO, evolution). No LLMs in the system.
Full plan: `docs/PLAN.md` — read the relevant phase section only when starting a new step or phase.

## Rules
1. Work only on the active step below. Don't expand scope; put ideas under Notes.
2. Stop and ask at major decisions: architecture, genome structure, recording format, new dependency, phase transition. Proceed without asking on small reversible details.
3. Verification output: short and raw — one number, count, or proof line. No full logs, no summaries.
4. After each step, update Status below: what finished, measured number, next step.
5. Experiments configured in `configs/` (YAML), fixed seeds, reproducible.
6. If training diverges (NaN, sim instability) or runs far longer than expected: record it and ask. Don't silently tweak hyperparameters.
7. No behavior or body is hand-scripted; only environments, tasks and rewards are defined.
8. When a phase's exit criterion is met: report the result, then before starting the next phase write a numbered step table for it into `docs/PLAN.md` (same format as Phase 1: step + one-line verification) and ask for approval. Phases 4–5 are only sketched; detail them together with the user first.

## Layout
`genome/` graph genome · `builder/` genome→MJCF · `envs/` procedural terrain + tasks · `train/` PPO inner loop · `evolve/` outer loop · `record/` trajectory export · `viewer/` Three.js · `runs/` (gitignored) · `configs/`

## Env
RTX 4060 8 GB, 32 GB RAM. JAX GPU needs Linux → WSL2 if on Windows. Check: `python -c "import jax; print(jax.devices())"`.

## Status
- Phase: 1
- Step: 1.1 (deps pinned in `requirements.txt`, `scripts/setup_env.sh`; GPU check pending on target machine)
- Last measurement: cloud container (no GPU): deps import OK, `jax.devices()` = `[CpuDevice(id=0)]`
- Notes: —
