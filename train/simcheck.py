"""Step 2b.1: sim validity. Re-evaluate a snapshot of each run under several sim settings (env_overrides)
and measure contact slip: tangential speed of the creature's material point at each floor contact.
Slip is computed on CPU (MuJoCo C, same model options) from the GPU rollout's qpos/qvel.
Usage: python -m train.simcheck configs/phase2b_simcheck.yaml"""
import os
import sys

import jax
import mujoco
import numpy as np
import yaml

from train.envs import load_configs
from train.rollout import body_stats, make_rollout, policy_act_fn
from train.snapshots import snapshot_paths
from envs.locomotion import Locomotion


def contact_stats(m, qpos, qvel):
    """qpos/qvel: [T, N, ...] -> (slip speeds m/s, penetration depths m) over all floor contacts."""
    d = mujoco.MjData(m)
    v6 = np.zeros(6)
    slip, pen = [], []
    for q, v in zip(qpos.reshape(-1, m.nq), qvel.reshape(-1, m.nv)):
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(v))):
            continue
        d.qpos[:], d.qvel[:] = q, v
        mujoco.mj_forward(m, d)
        for c in d.contact[:d.ncon]:
            g = c.geom2 if m.geom_bodyid[c.geom1] == 0 else c.geom1
            b = m.geom_bodyid[g]
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, v6, 0)  # [ang, lin] at xpos, world frame
            vp = v6[3:] + np.cross(v6[:3], c.pos - d.xpos[b])
            n = c.frame[:3]
            slip.append(np.linalg.norm(vp - vp.dot(n) * n))
            pen.append(max(-c.dist, 0.0))
    return np.array(slip), np.array(pen)


def check(cfg, sim_name, overrides):
    env = Locomotion(cfg["genome"], config_overrides=overrides or None)
    ckpt = snapshot_paths(cfg["run_dir"])[cfg["snapshot"]]
    run = make_rollout(env, policy_act_fn(ckpt), cfg["num_envs"], cfg["num_steps"], None, keep_poses=True)
    vx, poses = jax.device_get(run(cfg["seed"]))
    stats = " ".join(f"{k} {v:.2f}" for k, v in body_stats(env, vx, poses).items())
    k = cfg["slip_envs"]
    slip, pen = contact_stats(env.mj_model, poses["qpos"][:, :k], poses["qvel"][:, :k])
    p = lambda a, x: float(np.percentile(a, x)) if a.size else float("nan")
    print(f"{os.path.basename(os.path.dirname(cfg['run_dir']))} {cfg['env']} {sim_name} mean_vx {float(vx):.2f} "
          f"{stats} slip_p50 {p(slip, 50):.2f} slip_p90 {p(slip, 90):.2f} "
          f"pen_p90_mm {1000 * p(pen, 90):.1f} contacts/step {slip.size / (k * cfg['num_steps']):.1f}", flush=True)


def main(cfg_path):
    with open(cfg_path) as f:
        top = yaml.safe_load(f)
    for root in top["run_roots"]:
        for cfg in load_configs(cfg_path):
            cfg["run_dir"] = os.path.join(root, cfg["env"])
            for sim_name, overrides in top["sims"].items():
                check(cfg, sim_name, overrides)


if __name__ == "__main__":
    main(sys.argv[1])
