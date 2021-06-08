from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import jax
import functools
import numpy as np
import jax.numpy as jnp
from gym import utils
from gym.envs.mujoco import mujoco_env


class CartpoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    PENDULUM_LENGTH = 0.6
    ACTION_MAX = 3.0
    ACTION_MIN = -3.0

    def __init__(self):
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/cartpole.xml" % dir_path, 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        cost_lscale = CartpoleEnv.PENDULUM_LENGTH
        reward = np.exp(
            -np.sum(
                np.square(
                    self._get_ee_pos(ob) - np.array([0.0, CartpoleEnv.PENDULUM_LENGTH])
                )
            )
            / (cost_lscale ** 2)
        )
        reward -= 0.01 * np.sum(np.square(a))
        return self.obs_process(ob), reward, False, {}

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(0, 0.1, np.shape(self.init_qpos))
        qvel = self.init_qvel + np.random.normal(0, 0.1, np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(self):
        return self.obs_process(super().reset())

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    @staticmethod
    def _get_ee_pos(x):
        x0, theta = x[0], x[1]
        return np.array(
            [
                x0 - CartpoleEnv.PENDULUM_LENGTH * np.sin(theta),
                -CartpoleEnv.PENDULUM_LENGTH * np.cos(theta),
            ]
        )

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = 4

    #
    # - Hand Crafted Observation (https://github.com/quanvuong/handful-of-trials-pytorch)
    # - Reward Function (compute_reward)
    #

    def GT_Reward(self, obs, acs, obs_next):
        low = CartpoleEnv.ACTION_MIN
        high = CartpoleEnv.ACTION_MAX
        cost_lscale = CartpoleEnv.PENDULUM_LENGTH
        target = jnp.array([0.0, CartpoleEnv.PENDULUM_LENGTH])
        dist_2_goal = CartpoleEnv._get_ee_pos_from_proc_obs(obs_next) - target
        dist_2_goal = -(dist_2_goal ** 2).sum(-1)
        dist_2_goal = jnp.exp(dist_2_goal / (cost_lscale ** 2))
        denorm_acs = low + (high - low) * ((acs + 1) / (2))
        acs_cost = 0.01 * (denorm_acs ** 2).sum(axis=-1)
        reward = dist_2_goal - acs_cost
        return reward

    @jax.partial(jax.jit, static_argnums=(0,))
    def reward_fn(self, obs, acs, obs_next):
        cost_lscale = CartpoleEnv.PENDULUM_LENGTH
        target = jnp.array([0.0, CartpoleEnv.PENDULUM_LENGTH])
        dist_2_goal = CartpoleEnv._get_ee_pos_from_proc_obs(obs_next) - target
        dist_2_goal = -(dist_2_goal ** 2).sum(-1)
        dist_2_goal = jnp.exp(dist_2_goal / (cost_lscale ** 2))
        acs_cost = 0.01 * (acs ** 2).sum(axis=-1)
        reward = dist_2_goal - acs_cost
        return reward

    @staticmethod
    def _get_ee_pos_from_proc_obs(obs):
        sin, cos, x0 = obs[:1], obs[1:2], obs[2:3]
        return jnp.concatenate(
            [
                x0 - CartpoleEnv.PENDULUM_LENGTH * sin,
                -CartpoleEnv.PENDULUM_LENGTH * cos,
            ],
            -1,
        )

    @staticmethod
    def obs_process(obs):
        if len(obs.shape) == 1:
            # We assume it is a numpy array.
            return np.concatenate(
                [np.sin(obs[1:2]), np.cos(obs[1:2]), obs[:1], obs[2:]], axis=-1
            )

        if isinstance(obs, np.ndarray):
            return np.concatenate(
                [np.sin(obs[:, 1:2]), np.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]],
                axis=-1,
            )
        raise NotImplementedError()
