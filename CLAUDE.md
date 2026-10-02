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
RTX 4060 8 GB, 32 GB RAM. JAX GPU needs Linux → WSL2 (distro `Ubuntu-24.04`, repo at `~/RL-Project`, setup `scripts/setup_env.sh`). Check: `python -c "import jax; print(jax.devices())"`.

## Status
- Phase: 1 complete (user accepted exit 2026-10-02) → Phase 2 step table written in `docs/PLAN.md`, awaiting user approval before 2.1
- Step: 1.6 done: `record/export.py` → `runs/phase1_go1/recordings/end.json`, 13 parts (37 primitive geoms), 500 frames, fitness 0.91; spec `docs/RECORDING.md`
- Last measurement (2026-10-02, RTX 4060 Laptop, WSL2 Ubuntu-24.04, py3.11):
  - 1.1 `[CudaDevice(id=0)]`
  - 1.2 `FINAL Go1JoystickFlatTerrain step 206438400 mean_reward 28.27 time 1132s` (8192 envs fit, 6.6 GB)
  - 1.3 snapshots start/mid/end = ckpt 0 / 114688000 / 206438400 → `runs/phase1_go1/snapshots/`
  - 1.4 `configs/phase1_eval.yaml` (seed 1, 128 envs × 1000 steps, fixed cmd vx=1.0): mean_vx random −0.012, start −0.004, mid 0.918, end 0.928
  - 1.5 `runs/phase1_go1/videos/start_mid_end.mp4`
- Notes:
  - Exit criterion "end ≥ 5× random" is ill-posed: random/start ≈ 0 (slightly negative) → ratio −79. End tracks 93% of commanded speed; user accepted as passed. Phase 2 uses an absolute threshold instead.
  - Recording v1 stores primitive geoms only; Go1 visual meshes (~100k verts) skipped, its collision primitives stand in.
  - MJX-Warp printed "solver iterations limit reached" every step (Playground Go1 uses 1 solver iteration by design): 2 GB log in 8 min, and it throttled training. `train/envs.py:load_env` sets `opt._impl.warn_overflow=0` (print-only, physics unchanged) → 200M steps in 19 min.
  - Brax saves no step-0 checkpoint; `train_ppo.save_initial_checkpoint` saves the exact init params (ppo.train with num_timesteps=0, same seed).
  - Brax 0.14.2 `load_policy` KeyErrors on `mean_kernel_init_fn: null`; `train/rollout.py:load_policy` works around it.
  - Run steps: `python -m train.train_ppo configs/phase1_go1.yaml`, then `python -m train.{snapshots,rollout,render} configs/phase1_eval.yaml` (render: `MUJOCO_GL=egl`).
