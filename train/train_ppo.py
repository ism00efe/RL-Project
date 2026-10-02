"""Train a stock Playground env with Brax PPO. Usage: python -m train.train_ppo configs/phase1_go1.yaml"""
import functools
import os
import sys
import time

import yaml
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    name = cfg["env"]
    run_dir = os.path.abspath(cfg["run_dir"])
    os.makedirs(run_dir, exist_ok=True)

    env = registry.load(name)
    eval_env = registry.load(name)
    randomizer = registry.get_domain_randomizer(name)

    ppo_params = locomotion_params.brax_ppo_config(name)
    for k, v in (cfg.get("ppo_overrides") or {}).items():
        ppo_params[k] = v
    train_params = dict(ppo_params)
    net_cfg = train_params.pop("network_factory")
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_cfg)

    t0 = time.time()
    last = {}

    def progress(step, metrics):
        last.update(step=step, reward=float(metrics["eval/episode_reward"]))
        print(f"step {step} reward {last['reward']:.2f} t {time.time() - t0:.0f}s", flush=True)

    ppo.train(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=randomizer,
        network_factory=network_factory,
        progress_fn=progress,
        seed=cfg["seed"],
        save_checkpoint_path=os.path.join(run_dir, "ckpt"),
        **train_params,
    )
    print(f"FINAL {name} step {last['step']} mean_reward {last['reward']:.2f} time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
