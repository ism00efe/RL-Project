"""Train a Playground env or genome body(ies) with Brax PPO. Usage: python -m train.train_ppo configs/<run>.yaml"""
import functools
import os
import sys
import time

import jax
import jax.numpy as jnp
from brax.training.acme import specs
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

from train.envs import load_configs, make_env


def save_initial_checkpoint(ckpt_dir, env, train_kwargs):
    """Brax only checkpoints after training iterations; save the untrained params as step 0.
    num_timesteps=0 makes ppo.train return right after init, so with the same seed and
    config these are exactly the params the real run starts from."""
    _, params, _ = ppo.train(**{**train_kwargs, "num_timesteps": 0, "progress_fn": lambda *a: None})
    obs = jax.eval_shape(env.reset, jax.random.PRNGKey(0)).obs
    cfg = checkpoint.network_config(
        observation_size=jax.tree.map(lambda x: specs.Array(x.shape[-1:], jnp.float32), obs),
        action_size=env.action_size,
        normalize_observations=train_kwargs.get("normalize_observations", False),
        network_factory=train_kwargs["network_factory"],
    )
    checkpoint.save(ckpt_dir, 0, params, cfg)


def ppo_params(cfg):
    """Explicit `ppo` block (genome runs), else Playground's tuned config + `ppo_overrides`."""
    if "ppo" in cfg:
        return {**cfg["ppo"], "network_factory": dict(cfg["ppo"]["network_factory"])}
    params = locomotion_params.brax_ppo_config(cfg["env"])
    for k, v in (cfg.get("ppo_overrides") or {}).items():
        params[k] = v
    return dict(params)


def train(cfg):
    name = cfg["env"]
    run_dir = os.path.abspath(cfg["run_dir"])
    os.makedirs(run_dir, exist_ok=True)

    env = make_env(cfg)
    eval_env = make_env(cfg)
    randomizer = None if "genome" in cfg else registry.get_domain_randomizer(name)

    train_params = ppo_params(cfg)
    net_cfg = train_params.pop("network_factory")
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_cfg)

    t0 = time.time()
    last = {}

    def progress(step, metrics):
        last.update(step=step, reward=float(metrics["eval/episode_reward"]))
        print(f"step {step} reward {last['reward']:.2f} t {time.time() - t0:.0f}s", flush=True)

    train_kwargs = dict(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=randomizer,
        network_factory=network_factory,
        progress_fn=progress,
        seed=cfg["seed"],
        **train_params,
    )
    ckpt_dir = os.path.join(run_dir, "ckpt")
    save_initial_checkpoint(ckpt_dir, env, train_kwargs)
    ppo.train(save_checkpoint_path=ckpt_dir, **train_kwargs)
    print(f"FINAL {name} step {last['step']} mean_reward {last['reward']:.2f} time {time.time() - t0:.0f}s", flush=True)


def main(cfg_path):
    for cfg in load_configs(cfg_path):
        train(cfg)


if __name__ == "__main__":
    main(sys.argv[1])
