"""Step 1.4: fixed-seed rollouts of each snapshot (+ uniform-random actions) under a fixed command.
Prints mean forward velocity (body-frame x, m/s; steps after a fall count as 0).
Usage: python -m train.rollout configs/phase1_eval.yaml"""
import json
import os
import sys

import jax
import jax.numpy as jp
import yaml
from brax.training import checkpoint as brax_checkpoint
from brax.training import networks
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict

from train.envs import load_env
from train.snapshots import snapshot_paths

HOLD = 10**9  # steps_until_next_cmd: never resample the command
RENDER_FIELDS = ("qpos", "qvel", "mocap_pos", "mocap_quat", "xfrc_applied")


def fix_command(state, command):
    info = dict(state.info)
    info["command"] = jp.broadcast_to(command, info["command"].shape)
    info["steps_until_next_cmd"] = jp.full_like(info["steps_until_next_cmd"], HOLD)
    return state.replace(info=info)


def make_rollout(env, act_fn, num_envs, num_steps, command, keep_poses=False):
    """act_fn(obs, key) -> batched actions. Returns jitted fn(seed) -> (mean_vx, poses|None),
    poses = {field: [num_steps, num_envs, ...]} for RENDER_FIELDS of state.data."""
    command = jp.asarray(command, dtype=jp.float32)
    reset, step = jax.vmap(env.reset), jax.vmap(env.step)
    vx = jax.vmap(lambda d: env.get_local_linvel(d)[0])

    def body(carry, _):
        state, alive, key = carry
        key, k = jax.random.split(key)
        state = fix_command(step(state, act_fn(state.obs, k)), command)
        alive = alive * (1.0 - state.done)
        pose = {f: getattr(state.data, f) for f in RENDER_FIELDS} if keep_poses else None
        out = (vx(state.data) * alive, pose)
        return (state, alive, key), out

    @jax.jit
    def run(seed):
        key = jax.random.PRNGKey(seed)
        key, k = jax.random.split(key)
        state = fix_command(reset(jax.random.split(k, num_envs)), command)
        alive = jp.ones(num_envs)
        _, (v, poses) = jax.lax.scan(body, (state, alive, key), None, length=num_steps)
        return v.mean(), poses

    return run


def load_policy(ckpt_path, deterministic=True):
    """checkpoint.load_policy, except brax 0.14.2 saves None kernel-init fns
    (e.g. mean_kernel_init_fn) but its load_config KeyErrors on them; pass None through."""
    with open(os.path.join(ckpt_path, "ppo_network_config.json")) as f:
        raw = json.load(f)
    kw = raw["network_factory_kwargs"]
    kw["activation"] = networks.ACTIVATION[kw["activation"]]
    for k in brax_checkpoint._KERNEL_INIT_FN_KEYWORDS:
        if kw.get(k) is not None:
            kw[k] = networks.KERNEL_INITIALIZER[kw[k]]
    net = brax_checkpoint.get_network(config_dict.create(**raw), ppo_networks.make_ppo_networks)
    params = checkpoint.load(ckpt_path)
    return ppo_networks.make_inference_fn(net)(params, deterministic=deterministic)


def policy_act_fn(ckpt_path):
    policy = load_policy(ckpt_path)
    return lambda obs, key: policy(obs, key)[0]


def random_act_fn(action_size):
    return lambda obs, key: jax.random.uniform(
        key, (jax.tree.leaves(obs)[0].shape[0], action_size), minval=-1.0, maxval=1.0)


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    env = load_env(cfg["env"])
    acts = {"random": random_act_fn(env.action_size)}
    acts.update({n: policy_act_fn(p) for n, p in snapshot_paths(cfg["run_dir"]).items()})
    res = {}
    for name, act in acts.items():
        run = make_rollout(env, act, cfg["num_envs"], cfg["num_steps"], cfg["command"])
        res[name] = float(run(cfg["seed"])[0])
        print(f"{name} mean_vx {res[name]:.3f}", flush=True)
    print(f"end/start {res['end'] / res['start']:.1f}  end/random {res['end'] / res['random']:.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
