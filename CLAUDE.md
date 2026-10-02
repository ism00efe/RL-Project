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
- Phase: 2b (physics credibility), step 2b.1 next. User approved all of 2b (table in `docs/PLAN.md`) as one block on 2026-10-02: run 2b.1→2b.4 without asking per step; still stop for anything outside that table or rule-6 surprises. Phase 2 steps 2.1–2.8 done (results below).
- Last measurement (2026-10-02, RTX 4060 Laptop, WSL2):
  - 2.1 round-trip 4/4 · 2.2 parts/joints: quadruped 9/12, hexapod 13/18, snake 8/14, biped 7/8
  - 2.3 zero-action 10 s: 0 NaN all 4 · 2.4 env-steps/s 198k–236k (4096 envs)
  - 2.5 end mean_vx m/s, `configs/phase2_train.yaml` (base) → `phase2_train_upright.yaml`: quadruped 4.76→7.63, hexapod 4.57→9.07, snake 1.95→1.64, biped 2.26→5.48 (random/start ≈ 0)
  - 2.6 recordings `runs/phase2{,_upright}/<genome>/recordings/end.json` · 2.7/2.8 player https://claude.ai/artifact/Y4zQUNqsThGYzY7zaYMcrz (private)
- Notes:
  - Genome v1 `docs/GENOME.md` and reward (forward vel − 0.05·mean(a²), no termination) were chosen without explicit user review; user may veto.
  - User's view (2026-10-02): tumbling/odd gaits are fine if physically valid ("don't anthropomorphize"); energy = planned ecosystem feature, fine to add now; damage/health later (Phase 4). Hypotheses to test, not facts: equal "power" for all parts is unrealistic (muscles scale with cross-section), simulator physics may be the real cause.
  - Base reward: quadruped/hexapod learned to run upside down (upright 2%). `upright_termination` (opt-in env flag) fixes it (upright 100%), but speeds are implausible: biped hops/skates on one sliding foot at 5.5 m/s; contact slip ~2 m/s median. Likely causes: motor strength (gear 15–40 N·m on 0.5–1 kg links), tiny energy cost, soft contacts (4 solver iters). Candidates: torque·velocity energy cost, lower gear, more solver iterations / stiffer contacts. Needs user decision (rule 6).
  - Phase 1 (done): Go1 end mean_vx 0.928 at cmd 1.0; recording v1 `docs/RECORDING.md`; exit accepted by user.
  - MJX-Warp overflow printf disabled (`warn_overflow=0`, print-only) — it wrote 2 GB/8 min and throttled training.
  - Brax 0.14.2 checkpoint quirks handled in `train/rollout.py:load_policy`; no step-0 ckpt in brax → `train_ppo.save_initial_checkpoint`.
  - Run: `python -m train.train_ppo <train.yaml>`, then `python -m train.{snapshots,rollout,render} <eval.yaml>`, `python -m record.export <eval.yaml>` (`MUJOCO_GL=egl`). Configs with `genomes:` expand per body. Viewer: `python record/collect.py <rec.json[:tag]>...`, `npm --prefix viewer run build:artifact`.

## Operating on the user's machine (Windows host + WSL2)
- Canonical repo + venv + runs: WSL `Ubuntu-24.04`, `/root/RL-Project` (`.venv`, py3.11). Commands run as root: `wsl -d Ubuntu-24.04 -u root -- bash <script.sh>`.
  From Git Bash set `MSYS_NO_PATHCONV=1`, and put multi-line commands in a script file (inline quoting through wsl.exe breaks). Strip CRLF (`sed -i 's/$//'`) from scripts written on Windows.
- Long runs: start detached so they survive tool timeouts: PowerShell `Start-Process wsl.exe -ArgumentList '-d','Ubuntu-24.04','-u','root','--','bash','<script>' -WindowStyle Hidden`; log to a file inside WSL, never tee to stdout.
- GPU: one training at a time (JAX preallocates 75% of 8 GB). CPU-side checks: `JAX_PLATFORMS=cpu` (slow, contends with training). Render: `MUJOCO_GL=egl`.
- Speed reference: Go1 200M steps 19 min; genome bodies 100M steps 8–11 min each (8192 envs).
- Viewer: Node on Windows (`viewer/`, `npm --prefix viewer run build:artifact`); recordings are collected into `viewer/public/recordings` (gitignored). The player is a claude.ai artifact; republish to the same URL (pass it as `url`).
