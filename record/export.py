"""Step 1.6: export one fixed-seed episode of a policy snapshot as recording format v1 (docs/RECORDING.md).
Usage: python -m record.export configs/phase1_eval.yaml"""
import json
import os
import sys

import jax
import mujoco
import numpy as np
import yaml

from train.envs import load_env
from train.rollout import make_rollout, policy_act_fn
from train.snapshots import snapshot_paths

FORMAT, VERSION = "evo-recording", 1
PRIMITIVES = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
}
SIZE_DIMS = {"sphere": 1, "capsule": 2, "cylinder": 2, "ellipsoid": 3, "box": 3}


def r4(x):
    return np.round(np.asarray(x, dtype=np.float64), 4).tolist()


def part_list(m):
    """Parts = all bodies except world; geoms = primitive geoms in body-local frame."""
    parts = []
    for b in range(1, m.nbody):
        geoms = []
        for g in np.flatnonzero(m.geom_bodyid == b):
            kind = PRIMITIVES.get(int(m.geom_type[g]))
            if kind is None:
                continue
            geoms.append({"type": kind, "size": r4(m.geom_size[g][:SIZE_DIMS[kind]]),
                          "pos": r4(m.geom_pos[g]), "quat": r4(m.geom_quat[g]), "rgba": r4(m.geom_rgba[g])})
        parent = int(m.body_parentid[b])
        parts.append({"name": m.body(b).name, "parent": parent - 1 if parent > 0 else None, "geoms": geoms})
    return parts


def part_frames(m, poses):
    """World pose of every part per frame, recomputed on CPU from qpos (forward kinematics)."""
    d = mujoco.MjData(m)
    pos, quat = [], []
    for qpos in poses["qpos"][:, 0]:
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        pos.append(r4(d.xpos[1:]))
        quat.append(r4(d.xquat[1:]))
    return {"pos": pos, "quat": quat}


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    rec = cfg["record"]
    env = load_env(cfg["env"])
    src = snapshot_paths(cfg["run_dir"])[rec["snapshot"]]
    run = make_rollout(env, policy_act_fn(src), 1, rec["steps"], cfg["command"], keep_poses=True)
    fitness, poses = jax.device_get(run(cfg["seed"]))
    out = {
        "format": FORMAT,
        "version": VERSION,
        "meta": {**rec["meta"], "env": cfg["env"], "fitness": round(float(fitness), 4),
                 "fitness_name": "mean_vx", "seed": cfg["seed"],
                 "source": os.path.relpath(src, os.getcwd())},
        "dt": env.dt,
        "num_frames": rec["steps"],
        "parts": part_list(env.mj_model),
        "frames": part_frames(env.mj_model, poses),
    }
    path = os.path.join(cfg["run_dir"], "recordings", f"{rec['snapshot']}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{path} parts {len(out['parts'])}")


if __name__ == "__main__":
    main(sys.argv[1])
