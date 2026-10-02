"""Generic flat-ground locomotion env for any compiled genome (Playground MjxEnv API).

obs: qpos[2:] (root height, root quat, joint angles) + qvel (root lin/ang vel, joint vels)
reward: forward_weight * root world-x velocity - ctrl_cost_weight * mean(action^2)
        - energy_cost_weight * mechanical power / (mass * g)
motors: max torque from the muscle model (builder.compile.max_torque), force-velocity limited (max_joint_speed)
done: non-finite state; optionally (upright_termination) when the root's local z axis points down.
"""
import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from builder.compile import compile_genome
from genome.genome import load

# Worst-case contacts of one geom against the floor plane.
PLANE_CONTACTS = {mujoco.mjtGeom.mjGEOM_BOX: 4, mujoco.mjtGeom.mjGEOM_CAPSULE: 2, mujoco.mjtGeom.mjGEOM_SPHERE: 1}


def default_config():
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        action_repeat=1,
        forward_weight=1.0,
        ctrl_cost_weight=0.05,
        energy_cost_weight=0.0,  # x mechanical power / (m*g): reward = vx * (1 - w*CoT_mech)
        max_joint_speed=15.0,    # rad/s, force-velocity limit (None = off)
        upright_termination=False,  # end episode when root body is upside down (anti-flipping)
        reset_noise=0.05,       # uniform noise on joint angles (rad) and velocities at reset
        max_envs=8192,          # sizes the MJX-Warp contact buffer (shared across all envs)
        impl="warp",
        # Sim fidelity overrides (None = compiler default: 4 iterations, 8 ls_iterations, solref timeconst 0.02 s).
        solver_iterations=None,
        ls_iterations=None,
        contact_timeconst=None,  # geom solref[0] for all geoms (s); smaller = stiffer, needs >= 2*sim_dt
    )


class Locomotion(mjx_env.MjxEnv):
    def __init__(self, genome, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)
        self._genome = load(genome) if isinstance(genome, str) else genome
        self._xml, self._mj_model = compile_genome(self._genome)
        self._mj_model.opt.timestep = self._config.sim_dt
        if self._config.solver_iterations is not None:
            self._mj_model.opt.iterations = self._config.solver_iterations
        if self._config.ls_iterations is not None:
            self._mj_model.opt.ls_iterations = self._config.ls_iterations
        if self._config.contact_timeconst is not None:
            self._mj_model.geom_solref[:, 0] = self._config.contact_timeconst
        mx = mjx.put_model(self._mj_model, impl=self._config.impl)
        if hasattr(mx.opt._impl, "warn_overflow"):  # see train/envs.py
            mx = mx.replace(opt=mx.opt.replace(_impl=mx.opt._impl.replace(warn_overflow=0)))
        self._mjx_model = mx
        m = self._mj_model
        ncon = sum(PLANE_CONTACTS.get(int(t), 4) for t, b in zip(m.geom_type, m.geom_bodyid) if b > 0)
        self._naconmax = ncon * self._config.max_envs
        self._njmax = 2 * (m.njnt - 1) + 4 * ncon
        self._joint_qpos = jp.arange(7, m.nq)
        self._joint_qvel = jp.arange(6, m.nv)
        self._act_dof = jp.array(m.jnt_dofadr[m.actuator_trnid[:, 0]])
        self._gear = jp.array(m.actuator_gear[:, 0])
        self._mass = float(m.body_subtreemass[1])

    def reset(self, rng):
        rng, k1, k2 = jax.random.split(rng, 3)
        n = self._config.reset_noise
        qpos = jp.array(self._mj_model.qpos0)
        qpos = qpos.at[self._joint_qpos].add(jax.random.uniform(k1, (self._joint_qpos.size,), minval=-n, maxval=n))
        qvel = jax.random.uniform(k2, (self._mj_model.nv,), minval=-n, maxval=n)
        data = mjx_env.make_data(self._mj_model, qpos=qpos, qvel=qvel, impl=self._mjx_model.impl.value,
                                 naconmax=self._naconmax, njmax=self._njmax)
        data = mjx.forward(self._mjx_model, data)
        metrics = {"forward_vel": jp.zeros(()), "ctrl_cost": jp.zeros(()), "power": jp.zeros(())}
        return mjx_env.State(data, self._obs(data), jp.zeros(()), jp.zeros(()), metrics, {"rng": rng})

    def _substeps(self, data, action):
        """n_substeps of physics. Force-velocity limit: a motor's torque in its direction of motion
        falls linearly to 0 at max_joint_speed (no limit when resisting motion). Returns
        (data, mean mechanical power sum|tau*omega| in W over the control step)."""
        wmax = self._config.max_joint_speed

        def sub(carry, _):
            data, work = carry
            w = data.qvel[self._act_dof]
            ctrl = action if wmax is None else action * jp.clip(1.0 - jp.sign(action) * w / wmax, 0.0, 1.0)
            work = work + jp.sum(jp.abs(self._gear * ctrl * w))
            return (mjx.step(self._mjx_model, data.replace(ctrl=ctrl)), work), None

        (data, work), _ = jax.lax.scan(sub, (data, jp.zeros(())), (), self.n_substeps)
        return data, work / self.n_substeps

    def step(self, state, action):
        data, power = self._substeps(state.data, action)
        vx = data.qvel[0]
        ctrl_cost = jp.mean(jp.square(action))
        energy_cost = power / (self._mass * 9.81)  # W / N = m/s, same unit as vx
        reward = (self._config.forward_weight * vx - self._config.ctrl_cost_weight * ctrl_cost
                  - self._config.energy_cost_weight * energy_cost)
        ok = jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
        done = ~ok
        if self._config.upright_termination:
            done = done | (self.upright(data) < 0.0)
        done = done.astype(jp.float32)
        reward = jp.where(ok, reward, 0.0)
        state.metrics.update(forward_vel=jp.where(ok, vx, 0.0), ctrl_cost=ctrl_cost, power=jp.where(ok, power, 0.0))
        return state.replace(data=data, obs=self._obs(data), reward=reward, done=done)

    def _obs(self, data):
        obs = jp.concatenate([data.qpos[2:], data.qvel])
        return jp.where(jp.isfinite(obs), obs, 0.0)

    def upright(self, data):
        """World-z component of the root's local z axis (1 upright, -1 upside down)."""
        x, y = data.qpos[4], data.qpos[5]
        return 1.0 - 2.0 * (x * x + y * y)

    def forward_vel(self, data):
        """Fitness signal used by train.rollout: root velocity along world x."""
        return data.qvel[0]

    @property
    def xml_path(self):
        return f"<genome {self._genome['id']}>"

    @property
    def action_size(self):
        return self._mj_model.nu

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model

    @property
    def genome(self):
        return self._genome
