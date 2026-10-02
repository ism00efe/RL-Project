"""Steps 2.3/2.4: passive stability (zero action, 10 s, 16 envs) and throughput (random actions) per genome.
Usage: python -m envs.check genome/examples/*.json"""
import sys
import time

import jax
import jax.numpy as jp

from envs.locomotion import Locomotion

STABLE_ENVS, STABLE_STEPS = 16, 500
SPEED_ENVS, SPEED_STEPS = 4096, 200


def run(env, num_envs, num_steps, random_actions, seed=0):
    reset, step = jax.vmap(env.reset), jax.vmap(env.step)

    def body(carry, _):
        state, key = carry
        key, k = jax.random.split(key)
        act = (jax.random.uniform(k, (num_envs, env.action_size), minval=-1, maxval=1)
               if random_actions else jp.zeros((num_envs, env.action_size)))
        state = step(state, act)
        return (state, key), (jp.abs(state.data.qvel).max(), state.done.sum())

    @jax.jit
    def go(key):
        k1, k2 = jax.random.split(key)
        state = reset(jax.random.split(k1, num_envs))
        (state, _), (qv, nan) = jax.lax.scan(body, (state, k2), None, length=num_steps)
        return qv.max(), nan.sum(), state.data.qpos[:, 2].min()

    return go(jax.random.PRNGKey(seed))


def main(paths):
    for path in paths:
        env = Locomotion(path)
        name = env.genome["id"]
        qv, nan, z = jax.device_get(run(env, STABLE_ENVS, STABLE_STEPS, False))
        print(f"2.3 {name} nan_steps {int(nan)} max|qvel| {float(qv):.2f} final_min_root_z {float(z):.3f}", flush=True)
        jax.block_until_ready(run(env, SPEED_ENVS, SPEED_STEPS, True))  # compile
        t = time.time()
        _, nan, _ = jax.block_until_ready(run(env, SPEED_ENVS, SPEED_STEPS, True, seed=1))
        sps = SPEED_ENVS * SPEED_STEPS / (time.time() - t)
        print(f"2.4 {name} env-steps/s {sps:,.0f} nan_steps {int(nan)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
