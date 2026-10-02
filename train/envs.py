"""Shared env loading for train/eval scripts."""
import os

import yaml
from mujoco_playground import registry


def load_env(name):
    """registry.load, with MJX-Warp overflow printf warnings off.

    Playground's Go1 config runs 1 solver iteration by design, so the warp kernels print
    "solver iterations limit reached" every step (~250 MB/min of log). warn_overflow only
    gates those prints; simulation is unchanged.
    """
    env = registry.load(name)
    m = getattr(env, "_mjx_model", None)
    if m is not None and hasattr(m.opt._impl, "warn_overflow"):
        env._mjx_model = m.replace(opt=m.opt.replace(_impl=m.opt._impl.replace(warn_overflow=0)))
    return env


def make_env(cfg):
    """Genome locomotion env if cfg has `genome`, else the Playground env named by `env`."""
    if "genome" in cfg:
        from envs.locomotion import Locomotion
        return Locomotion(cfg["genome"], config_overrides=cfg.get("env_overrides") or None)
    return load_env(cfg["env"])


def load_configs(cfg_path):
    """One config per run. A config with `genomes: [paths]` + `run_root` expands to one run per
    genome: genome=<path>, env=<genome id>, run_dir=<run_root>/<genome id>."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if "genomes" not in cfg:
        return [cfg]
    runs = []
    for path in cfg["genomes"]:
        gid = os.path.splitext(os.path.basename(path))[0]
        run = {k: v for k, v in cfg.items() if k not in ("genomes", "run_root")}
        runs.append({**run, "genome": path, "env": gid, "run_dir": os.path.join(cfg["run_root"], gid)})
    return runs
