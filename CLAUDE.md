# evo — agent brief

Co-evolution of body + controller for 3D creatures (MuJoCo/MJX, PPO, evolution). No LLMs in the system.
Full plan: `docs/PLAN.md` (v2, built from the final ecosystem goal) — read its principles before proposing anything; read the relevant stage section when starting it.

## Rules
1. Work only on the active step below. Don't expand scope; put ideas under Notes.
2. Stop and ask at major decisions: architecture, genome structure, recording format, new dependency, phase transition. Proceed without asking on small reversible details.
3. Verification output: short and raw — one number, count, or proof line. No full logs, no summaries.
4. After each step, update Status below: what finished, measured number, next step.
5. Experiments configured in `configs/` (YAML), fixed seeds, reproducible.
6. If training diverges (NaN, sim instability) or runs far longer than expected: record it and ask. Don't silently tweak hyperparameters.
7. No behavior or body is hand-scripted; only the world is defined (terrain, physics, food, energy and life-cycle rules). No external fitness or speed goal (see PLAN principles).
8. When a phase's exit criterion is met: report the result, then before starting the next phase write a numbered step table for it into `docs/PLAN.md` (same format as Phase 1: step + one-line verification) and ask for approval. Stages beyond the next one are only sketched; detail them together with the user first. Every stage must add a piece of the final world.

## Communicating with the user
Turkish, short and clear. Lead with the problem or result; few numbers, after the point. Answer each question separately and directly.
Say why you ask something. Where the user must decide: give options + your recommendation. Reason from the final goal (PLAN),
never treat an intermediate metric or a feasibility test as the goal.

## Layout
`genome/` graph genome · `builder/` genome→MJCF · `envs/` procedural terrain + tasks · `train/` PPO inner loop · `evolve/` outer loop · `record/` trajectory export · `viewer/` Three.js · `runs/` (gitignored) · `configs/`

## Env
RTX 4060 8 GB, 32 GB RAM. JAX GPU needs Linux → WSL2 (distro `Ubuntu-24.04`, repo at `~/RL-Project`, setup `scripts/setup_env.sh`). Check: `python -c "import jax; print(jax.devices())"`.

## Status
- Stage: Plan v2 (`docs/PLAN.md`) written and reviewed with user 2026-10-03 (decisions recorded in PLAN "Decisions"). Foundations F1–F3 done. E1 step table drafted in PLAN (E1.1–E1.10, 2026-10-03). Decided: E1 test world = food arena (PLAN "Decision A").
  Next: user approves the table + template size (L×S) + `evosax`, confirms principle-1 wording; then start E1.1.
- Core decisions (user, 2026-10-02/03): the world selects (energy, death, reproduction), no external fitness or speed goal; mutations are random; brains inherited (neuroevolution), lifetime learning only if it wins an equal-compute comparison; standing still is a legitimate strategy; final goal = multi-biome ecosystem with interacting species (details in PLAN principles). Old "Goal gap" discussion: speed/forward-velocity came from the v1 plan's benchmark default, never from the user.
- Last measurement (2026-10-02, RTX 4060 Laptop, WSL2):
  - 2.1 round-trip 4/4 · 2.2 parts/joints: quadruped 9/12, hexapod 13/18, snake 8/14, biped 7/8
  - 2.3 zero-action 10 s: 0 NaN all 4 · 2.4 env-steps/s 198k–236k (4096 envs)
  - 2.5 end mean_vx m/s, `configs/phase2_train.yaml` (base) → `phase2_train_upright.yaml`: quadruped 4.76→7.63, hexapod 4.57→9.07, snake 1.95→1.64, biped 2.26→5.48 (random/start ≈ 0)
  - 2.6 recordings `runs/phase2{,_upright}/<genome>/recordings/end.json` · 2.7/2.8 player https://claude.ai/artifact/Y4zQUNqsThGYzY7zaYMcrz (private)
  - 2b.1 `python -m train.simcheck configs/phase2b_simcheck.yaml` (log `runs/phase2b_simcheck.log`), end snapshots, mean_vx default→strict (slip p50 m/s default→strict):
    base: quadruped 4.76→2.87 (1.11→0.35), hexapod 4.57→3.42 (0.79→0.28), snake 1.95→0.95 (0.97→0.19), biped 2.17→2.06 (0.75→0.00)
    upright: quadruped 7.62→1.75 (1.77→0.00), hexapod 9.07→5.99 (2.09→0.15), snake 1.63→0.67 (1.17→0.10), biped 4.87→3.84 (0.00→0.52)
    → default sim (dt 4 ms, 4 iters, solref 0.02) is exploited: speeds drop 5–77%, contact penetration p90 10–27 mm vs 1–6 mm. Strict = dt 1 ms, 20/20 iters, solref 0.005. Strict costs ~2× throughput only.
    Trained bodies are airborne most of the time (0.3–2 floor contacts/step; resting snake has 32).
  - 2b.2 muscle model, max torque N·m: quadruped/hexapod thigh 12.7 shin 7.4 · biped thigh 30.2 shin 20.2 foot 9.9 · snake seg 28.1 (was 5); zero-action 0 NaN all 4. FV limit max_joint_speed 15 rad/s.
  - 2b.3 energy cost `energy_cost_weight`·Σ|τω|/(m·g) (=1.0 in 2b configs); CoT J/(kg·m) + Fr printed by `train.rollout` / `train.simcheck`.
  - 2b.4 `configs/phase2b_train.yaml` (strict sim, w=1.0, no termination), 100M steps, 27–32 min/body; eval `phase2b_eval.yaml`, exit check `phase2b_simcheck_final.yaml` (log `runs/phase2b_chain.log`).
    end mean_vx m/s (default/strict/ultra) · upright · slip p50 · CoT J/(kg·m) · Fr: quadruped 0.02/0.02/0.02 · 1.00 · 0.00 · 2.2 · 0.00; hexapod 0.03/0.03/0.03 · 0.98 · 0.00 · 3.3 · 0.00;
    snake −0.00/−0.00/0.00 · 1.00 · 0.00 · n/a (no progress) · 0; biped 0.03/0.03/0.03 · 0.99 · 0.00 · 6.6 · 0.00. Final train reward: 15.8, 0.7, −54, −70.
    Physics is now sim-invariant (slip 0, Δ<20%), but every body is ~stationary (shuffles 2–3 cm/s): exit criterion met only trivially. Player updated (tag `2b`).
- Notes:
  - Since 2b.2 genome `strength` is a dimensionless muscle multiplier (default 1); Phase 2 configs/runs reproduce only at commit 619fcb2.
  - Genome v1 `docs/GENOME.md` and reward (forward vel − 0.05·mean(a²), no termination) were chosen without explicit user review; user may veto.
  - User's view (2026-10-02): tumbling/odd gaits are fine if physically valid ("don't anthropomorphize"); energy = planned ecosystem feature, fine to add now; damage/health later (Phase 4). Hypotheses to test, not facts: equal "power" for all parts is unrealistic (muscles scale with cross-section), simulator physics may be the real cause.
  - Base reward: quadruped/hexapod learned to run upside down (upright 2%). `upright_termination` (opt-in env flag) fixes it (upright 100%), but speeds are implausible: biped hops/skates on one sliding foot at 5.5 m/s; contact slip ~2 m/s median. Likely causes: motor strength (gear 15–40 N·m on 0.5–1 kg links), tiny energy cost, soft contacts (4 solver iters). Candidates: torque·velocity energy cost, lower gear, more solver iterations / stiffer contacts. Needs user decision (rule 6).
  - Phase 1 (done): Go1 end mean_vx 0.928 at cmd 1.0; recording v1 `docs/RECORDING.md`; exit accepted by user.
  - MJX-Warp overflow printf disabled (`warn_overflow=0`, print-only) — it wrote 2 GB/8 min and throttled training.
  - Brax 0.14.2 checkpoint quirks handled in `train/rollout.py:load_policy`; no step-0 ckpt in brax → `train_ppo.save_initial_checkpoint`.
  - Run: `python -m train.train_ppo <train.yaml>`, then `python -m train.{snapshots,rollout,render} <eval.yaml>`, `python -m record.export <eval.yaml>` (`MUJOCO_GL=egl`). Configs with `genomes:` expand per body. Viewer: `python record/collect.py <rec.json[:tag]>...`, `npm --prefix viewer run build:artifact`.

## Operating on the user's machine (Windows host + WSL2)
- Canonical repo + venv + runs: WSL `Ubuntu-24.04`, `/root/RL-Project` (`.venv`, py3.11). Commands run as root: `wsl -d Ubuntu-24.04 -u root -- bash <script.sh>`.
  From Git Bash set `MSYS_NO_PATHCONV=1`, and put multi-line commands in a script file (inline quoting through wsl.exe breaks). Strip CRLF (`sed -i 's/
$//'`) from scripts written on Windows.
- Long runs: start detached so they survive tool timeouts: PowerShell `Start-Process wsl.exe -ArgumentList '-d','Ubuntu-24.04','-u','root','--','bash','<script>' -WindowStyle Hidden`; log to a file inside WSL, never tee to stdout.
- GPU: one training at a time (JAX preallocates 75% of 8 GB). CPU-side checks: `JAX_PLATFORMS=cpu` (slow, contends with training). Render: `MUJOCO_GL=egl`.
- Speed reference: Go1 200M steps 19 min; genome bodies 100M steps 8–11 min each (8192 envs).
- Viewer: Node on Windows (`viewer/`, `npm --prefix viewer run build:artifact`); recordings are collected into `viewer/public/recordings` (gitignored). The player is a claude.ai artifact; republish to the same URL (pass it as `url`).
