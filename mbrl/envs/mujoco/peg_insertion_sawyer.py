import os
import jax
import numpy as np
import jax.numpy as jnp
from gym import utils
from mujoco_py import MjViewer
from mjrl.envs.mujoco_env import MujocoEnv

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join(BASE_PATH, "assets", "peg_insertion.xml")


class PegEnv(MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.peg_sid = -2
        self.target_sid = -1
        self.goal_y = 0.29
        self.init_target_pos = np.zeros(3)
        MujocoEnv.__init__(self, ASSET_PATH, 4)
        utils.EzPickle.__init__(self)
        self.peg_sid = self.model.site_name2id("peg_bottom")
        self.target_sid = self.model.site_name2id("target")
        self.init_body_pos = self.model.body_pos.copy()
        self.init_target_pos = self.data.site_xpos[self.target_sid]

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        obs = self.get_obs()
        reward = self.get_reward(obs, a)
        return obs, reward, False, self.get_env_infos()

    def get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat,
                self.data.qvel.ravel() * self.dt,
                self.data.site_xpos[self.peg_sid],
                [self.data.site_xpos[self.target_sid][1]],  # goal_y
            ]
        )

    def get_reward(self, obs, act=None):
        obs = np.clip(obs, -10.0, 10.0)
        if len(obs.shape) == 1:
            # vector obs, called when stepping the env
            hand_pos = obs[-4:-1]
            target_pos = self.goal2target(obs[-1])
            l1_dist = np.sum(np.abs(hand_pos - target_pos))
            l2_dist = np.linalg.norm(hand_pos - target_pos)
        else:
            obs = np.expand_dims(obs, axis=0) if len(obs.shape) == 2 else obs
            hand_pos = obs[:, :, -4:-1]
            target_pos = self.goal2target(obs[:, :, -1])
            l1_dist = np.sum(np.abs(hand_pos - target_pos), axis=-1)
            l2_dist = np.linalg.norm(hand_pos - target_pos, axis=-1)
        bonus = 5.0 * (l2_dist < 0.06)
        reward = -l1_dist - 5.0 * l2_dist + bonus
        return reward

    @jax.partial(jax.jit, static_argnums=(0,))
    def reward_fn(self, obs, act=None, obs_next=None):
        hand_pos = obs_next[-4:-1]
        goal_y = obs[-1]
        target_pos = jnp.array(
            [self.init_target_pos[0], goal_y, self.init_target_pos[2]]
        )
        l1_dist = jnp.sum(jnp.abs(hand_pos - target_pos))
        l2_dist = jnp.linalg.norm(hand_pos - target_pos)
        bonus = 5.0 * (l2_dist < 0.06)
        reward = -l1_dist - 5.0 * l2_dist + bonus
        return reward

    def compute_path_rewards(self, paths):
        # path has two keys: observations and actions
        # path["observations"] : (num_traj, horizon, obs_dim)
        # path["rewards"] should have shape (num_traj, horizon)
        obs = paths["observations"]
        rewards = self.get_reward(obs)
        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()

    # --------------------------------
    # resets and randomization
    # --------------------------------

    def robot_reset(self):
        self.set_state(self.init_qpos, self.init_qvel)

    def target_reset(self):
        # Randomize goal position
        self.goal_y = self.np_random.uniform(low=0.1, high=0.5)
        try:
            self.model.body_pos[-1, 1] = self.init_body_pos[-1, 1] + (
                self.goal_y - 0.29
            )
            self.model.body_pos[-2, 1] = self.init_body_pos[-2, 1] + (
                self.goal_y - 0.29
            )
            self.model.body_pos[-3, 1] = self.init_body_pos[-3, 1] + (
                self.goal_y - 0.29
            )
            self.sim.forward()
        except:
            pass

    def reset_model(self, seed=None):
        if seed is not None:
            self.seeding = True
            self.seed(seed)
        self.robot_reset()
        self.target_reset()
        return self.get_obs()

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        target_pos = self.model.body_pos[-1].copy()
        return dict(
            qp=self.data.qpos.copy(), qv=self.data.qvel.copy(), target_pos=target_pos
        )

    def set_env_state(self, state):
        self.sim.reset()
        qp = state["qp"].copy()
        qv = state["qv"].copy()
        target_pos = state["target_pos"]
        self.model.body_pos[-1] = target_pos
        self.goal_y = target_pos[1]
        self.data.qpos[:] = qp
        self.data.qvel[:] = qv
        self.model.body_pos[-1, 1] = self.init_body_pos[-1, 1] + (self.goal_y - 0.29)
        self.model.body_pos[-2, 1] = self.init_body_pos[-2, 1] + (self.goal_y - 0.29)
        self.model.body_pos[-3, 1] = self.init_body_pos[-3, 1] + (self.goal_y - 0.29)
        self.sim.forward()

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        return dict(state=self.get_env_state())

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth += 200
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent * 2.0

    def goal2target(self, goal_y):
        if type(goal_y) != np.ndarray:
            target_pos = np.array(
                [self.init_target_pos[0], goal_y, self.init_target_pos[2]]
            )
        else:
            assert len(goal_y.shape) == 2
            num_traj, horizon = goal_y.shape
            target_pos = np.zeros((num_traj, horizon, 3))
            target_pos[:, :, 0] = self.init_target_pos[0]
            target_pos[:, :, 1] = goal_y
            target_pos[:, :, 2] = self.init_target_pos[2]
        return target_pos
