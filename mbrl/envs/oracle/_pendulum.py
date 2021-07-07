# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from os import path

env_params = dict(
    g=10.0,
    m=1.0,
    ell=1.0,
    dt=0.05,
    state_size=2,
    action_size=1,
    n=2,
    max_torque=2,
    min_torque=-2,
    max_speed=8.0,
    max_episode_steps=200,
)

env_params['action_min'] = env_params['min_torque']
env_params['action_max'] = env_params['max_torque']


def angle_normalize(x):
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


def get_obs_pendulum(state):
    """ Return angle in polar coordinates and change. """
    th, thdot = state[0], state[1]
    return jnp.array([jnp.cos(th), jnp.sin(th), thdot])


def dynamic(state, action, env_params):
    th, thdot = state
    action = jnp.clip(action, env_params["min_torque"], env_params["max_torque"])
    newthdot = (
        thdot
        + (
            -3 * env_params["g"] / (2 * env_params["ell"]) * jnp.sin(th + jnp.pi)
            + 3.0 / (env_params["m"] * env_params["ell"] ** 2) * action
        )
        * env_params["dt"]
    )
    newth = th + newthdot * env_params["dt"]
    newthdot = jnp.clip(newthdot, -env_params["max_speed"], env_params["max_speed"])
    return jnp.reshape(jnp.array([newth, newthdot]), (2,))


def reward_fn(observation, u, env_params):
    th = jnp.arctan2(observation[1], observation[0])
    thdot = observation[2]
    u = jnp.clip(u, -env_params["max_torque"], env_params["max_torque"])
    costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
    return -costs[0].squeeze()


def reset_fn(rng, env_params):
    rng_th, rng_thdot = jax.random.split(rng, 2)
    th = jax.random.uniform(rng_th, minval=-jnp.pi, maxval=jnp.pi)
    thdot = jax.random.uniform(rng_thdot, minval=-1.0, maxval=1.0)
    state = jnp.array([th, thdot])
    return state, get_obs_pendulum(state)


def step_fn(state, u, env_params):
    """ Integrate pendulum ODE and return transition. """
    th, thdot = state[0], state[1]
    u = jnp.clip(u, -env_params["max_torque"], env_params["max_torque"])
    costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
    state = dynamic(state, u, env_params)
    return state, get_obs_pendulum(state), -costs[0], False, {}


def render(state, kwargs, mode="human"):
    viewer = kwargs.get("viewer", None)
    last_u = kwargs.get("last_u", None)
    if viewer is None:
        from gym.envs.classic_control import rendering

        viewer = rendering.Viewer(500, 500)
        viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
        rod = rendering.make_capsule(1, 0.2)
        rod.set_color(0.8, 0.3, 0.3)
        pole_transform = rendering.Transform()
        rod.add_attr(pole_transform)
        viewer.add_geom(rod)
        axle = rendering.make_circle(0.05)
        axle.set_color(0, 0, 0)
        viewer.add_geom(axle)
        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = rendering.Image(fname, 1.0, 1.0)
        imgtrans = rendering.Transform()
        img.add_attr(imgtrans)
        kwargs["pole_transform"] = pole_transform
        kwargs["viewer"] = viewer
        kwargs["img"] = img

    pole_transform = kwargs["pole_transform"]
    viewer.add_onetime(kwargs["img"])
    pole_transform.set_rotation(state[0] + np.pi / 2)
    if last_u:
        imgtrans.scale = (-last_u / 2, np.abs(last_u) / 2)

    return viewer.render(return_rgb_array=mode == "rgb_array"), kwargs
