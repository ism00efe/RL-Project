"""Step 1.3: pick start/mid/end checkpoints of a run and link them under <run_dir>/snapshots.
Usage: python -m train.snapshots configs/phase1_eval.yaml"""
import os
import sys

from train.envs import load_configs

NAMES = ("start", "mid", "end")


def select(run_dir):
    ckpt_dir = os.path.join(run_dir, "ckpt")
    steps = sorted(d for d in os.listdir(ckpt_dir) if d.isdigit())
    if len(steps) < 3:
        raise ValueError(f"need >= 3 checkpoints in {ckpt_dir}, found {len(steps)}")
    picks = (steps[0], steps[len(steps) // 2], steps[-1])
    return {n: os.path.join(ckpt_dir, s) for n, s in zip(NAMES, picks)}


def snapshot_paths(run_dir):
    return {n: os.path.abspath(os.path.join(run_dir, "snapshots", n)) for n in NAMES}


def link(cfg):
    run_dir = os.path.abspath(cfg["run_dir"])
    os.makedirs(os.path.join(run_dir, "snapshots"), exist_ok=True)
    for name, src in select(run_dir).items():
        dst = os.path.join(run_dir, "snapshots", name)
        if os.path.islink(dst):
            os.remove(dst)
        os.symlink(src, dst)
        print(f"{cfg['env']} {name} -> {os.path.basename(src)}")


def main(cfg_path):
    for cfg in load_configs(cfg_path):
        link(cfg)


if __name__ == "__main__":
    main(sys.argv[1])
