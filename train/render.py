"""Step 1.5: render one fixed-seed episode per snapshot and combine start|mid|end side by side.
Needs ffmpeg on PATH; set MUJOCO_GL=egl (or osmesa). Usage: python -m train.render configs/phase1_eval.yaml"""
import os
import subprocess
import sys
from types import SimpleNamespace

import jax
import numpy as np

from train.envs import load_configs, make_env
from train.rollout import make_rollout, policy_act_fn
from train.snapshots import NAMES, snapshot_paths


def write_mp4(path, frames, fps):
    h, w, _ = frames[0].shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, input=np.stack(frames).astype(np.uint8).tobytes(), check=True)


def render(cfg):
    r = cfg["render"]
    env = make_env(cfg)
    fps = round(1.0 / env.dt)
    out_dir = os.path.join(cfg["run_dir"], "videos")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, ckpt in snapshot_paths(cfg["run_dir"]).items():
        run = make_rollout(env, policy_act_fn(ckpt), 1, r["steps"], cfg.get("command"), keep_poses=True)
        poses = jax.device_get(run(cfg["seed"])[1])
        traj = [SimpleNamespace(data=SimpleNamespace(**{k: v[t, 0] for k, v in poses.items()}))
                for t in range(r["steps"])]
        frames = env.render(traj, height=r["height"], width=r["width"], camera=r["camera"])
        paths.append(os.path.join(out_dir, f"{name}.mp4"))
        write_mp4(paths[-1], frames, fps)
    out = os.path.join(out_dir, "start_mid_end.mp4")
    inputs = sum((["-i", p] for p in paths), [])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs,
                    "-filter_complex", f"hstack=inputs={len(NAMES)}", out], check=True)
    print(out, flush=True)


def main(cfg_path):
    for cfg in load_configs(cfg_path):
        render(cfg)


if __name__ == "__main__":
    main(sys.argv[1])
