"""Shared env loading for train/eval scripts."""
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
