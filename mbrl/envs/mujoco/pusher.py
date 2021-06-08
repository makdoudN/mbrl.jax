from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import jax
import jax.numpy as jnp
from gym import utils
from gym.envs.mujoco import mujoco_env


class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    ACTION_MIN = -2.0
    ACTION_MAX = 2.0

    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/pusher.xml" % dir_path, 4)
        utils.EzPickle.__init__(self)
        self.reset_model()

    def step(self, a):
        obj_pos = (self.get_body_com("object"),)
        vec_1 = obj_pos - self.get_body_com("tips_arm")
        vec_2 = obj_pos - self.get_body_com("goal")

        reward_near = -np.sum(np.abs(vec_1))
        reward_dist = -np.sum(np.abs(vec_2))
        reward_ctrl = -np.square(a).sum()
        reward = 1.25 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {"goal": self.get_body_com("goal")}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        self.cylinder_pos = np.array([-0.25, 0.15]) + np.random.normal(0, 0.025, [2])

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        self.ac_goal_pos = self.get_body_com("goal")

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[:7],
                self.sim.data.qvel.flat[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ]
        )

    # goal = ob[-3:]

    @jax.partial(jax.jit, static_argnums=(0,))
    def reward_fn(self, obs, acs, obs_next):
        qpos, qvel, tips_arm, obj_pos, goal_pos = jnp.split(
            obs_next, [7, 14, 17, 20], axis=-1
        )
        vec_1 = obj_pos - tips_arm
        vec_2 = obj_pos - goal_pos

        reward_near = -jnp.sum(jnp.abs(vec_1), axis=-1)
        reward_dist = -jnp.sum(jnp.abs(vec_2), axis=-1)
        reward_ctrl = -jnp.square(acs).sum(axis=-1)
        reward = 1.25 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        return reward
