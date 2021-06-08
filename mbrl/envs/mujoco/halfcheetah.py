# Copy: https://github.com/openai/gym/blob/master/gym/envs/mujoco/half_cheetah_v3.py

import functools
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import jax
import jax.numpy as jnp


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    CTRL_COST_WEIGHT = 0.1
    FORWARD_REWARD_WEIGHT = 1.0
    DT = 0.05

    def __init__(
        self,
        xml_file="half_cheetah.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
    ):
        utils.EzPickle.__init__(**locals())
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        self.prev_qpos = None

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)

        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }
        return self.obs_process(observation), reward, done, info

    def reset(self):
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        return self.obs_process(super().reset())

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ]
        )

    #
    # User modified function to leverage batched MBRL.
    #

    @staticmethod
    def obs_process(obs):
        return obs
        if isinstance(obs, np.ndarray):
            if obs.ndim == 1:
                return np.concatenate(
                    [obs[1:2], np.sin(obs[2:3]), np.cos(obs[2:3]), obs[3:]], axis=-1
                )
            else:
                return np.concatenate(
                    [
                        obs[..., 1:2],
                        np.sin(obs[..., 2:3]),
                        np.cos(obs[..., 2:3]),
                        obs[..., 3:],
                    ],
                    axis=1,
                )

    @jax.partial(jax.jit, static_argnums=(0,))
    def reward_fn(self, obs, acs, obs_next):
        control_cost = self.CTRL_COST_WEIGHT * jnp.sum(jnp.square(acs), -1)
        forward_cost = self.FORWARD_REWARD_WEIGHT * ((obs_next[0] - obs[0]))
        forward_cost = forward_cost / self.DT
        reward = forward_cost - control_cost
        return reward
