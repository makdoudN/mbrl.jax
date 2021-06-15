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

# NOTE: The observation is the state in cartpole.

import math

import gym
import jax
import jax.numpy as jnp
import numpy as np

from deluca.envs.core import Env
from deluca.utils import Random


env_params = dict(
    force_mag=10.0,
    gravity=9.8,
    masscart=1.0,
    masspole=0.1,
    length=0.5,  # actually half the pole's length
    tau=0.02,  # seconds between state updates
    kinematics_integrator="euler",
    # Angle at which to fail the episode
    # Angle at which to fail the episode
    theta_threshold_radians=12 * 2 * math.pi / 360,
    x_threshold=2.4,
    state_size=4,
    action_size=2,
    action_type="discrete",
    max_episode_steps=200,
)

env_params["total_mass"] = env_params["masspole"] + env_params["masspole"]
env_params["polemass_length"] = env_params["masspole"] * env_params["length"]
env_params["high"] = jnp.array(
    [
        env_params["x_threshold"] * 2,
        jnp.finfo(np.float32).max,
        env_params["theta_threshold_radians"] * 2,
        jnp.finfo(np.float32).max,
    ],
    dtype=jnp.float32,
)


def reset_fn(rng, env_params=None):
    state = jax.random.uniform(rng, shape=(4,), minval=-0.05, maxval=0.05)
    return state, state


def dynamics_fn(state, action, env_params):
    x, x_dot, theta, theta_dot = state
    force = jax.lax.cond(
        action == 1, lambda x: x, lambda x: -x, env_params["force_mag"]
    )
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)
    temp = (
        force + env_params["polemass_length"] * theta_dot ** 2 * sintheta
    ) / env_params["total_mass"]
    thetaacc = (env_params["gravity"] * sintheta - costheta * temp) / (
        env_params["length"]
        * (
            4.0 / 3.0
            - env_params["masspole"] * costheta ** 2 / env_params["total_mass"]
        )
    )
    xacc = (
        temp
        - env_params["polemass_length"] * thetaacc * costheta / env_params["total_mass"]
    )
    if env_params["kinematics_integrator"] == "euler":
        x = x + env_params["tau"] * x_dot
        x_dot = x_dot + env_params["tau"] * xacc
        theta = theta + env_params["tau"] * theta_dot
        theta_dot = theta_dot + env_params["tau"] * thetaacc
    else:  # semi-implicit euler
        x_dot = x_dot + env_params["tau"] * xacc
        x = x + env_params["tau"] * x_dot
        theta_dot = theta_dot + env_params["tau"] * thetaacc
        theta = theta + env_params["tau"] * theta_dot
    return jnp.array([x, x_dot, theta, theta_dot])


def terminal_fn(state, env_params):
    x, x_dot, theta, theta_dot = state
    done = jax.lax.cond(
        (jnp.abs(x) > jnp.abs(env_params["x_threshold"]))
        + (jnp.abs(theta) > jnp.abs(env_params["theta_threshold_radians"])),
        lambda done: True,
        lambda done: False,
        None,
    )
    return done


def reward_fn(state, env_params):
    return 1.0 - terminal(state, env_params)


def step_fn(state, action, env_params):
    state = dynamics_fn(state, action, env_params)
    done = terminal_fn(state, env_params)
    reward = 1.0 - done
    return state, state, reward, done, {}


def render(
    state,
    mode="human",
    viewer=None,
    pole=None,
    env_params=env_params,
    carttrans=None,
    poletrans=None,
):
    screen_width = 600
    screen_height = 400

    world_width = env_params["x_threshold"] * 2
    scale = screen_width / world_width
    carty = 100  # TOP OF CART
    polewidth = 10.0
    polelen = scale * (2 * env_params["length"])
    cartwidth = 50.0
    cartheight = 30.0

    if viewer is None:
        from gym.envs.classic_control import rendering

        viewer = rendering.Viewer(screen_width, screen_height)
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        carttrans = rendering.Transform()
        cart.add_attr(carttrans)
        viewer.add_geom(cart)
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        pole.set_color(0.8, 0.6, 0.4)
        poletrans = rendering.Transform(translation=(0, axleoffset))
        pole.add_attr(poletrans)
        pole.add_attr(carttrans)
        viewer.add_geom(pole)
        axle = rendering.make_circle(polewidth / 2)
        axle.add_attr(poletrans)
        axle.add_attr(carttrans)
        axle.set_color(0.5, 0.5, 0.8)
        viewer.add_geom(axle)
        track = rendering.Line((0, carty), (screen_width, carty))
        track.set_color(0, 0, 0)
        viewer.add_geom(track)

    if state is None:
        return None

    # Edit the pole polygon vertex
    l, r, t, b = (
        -polewidth / 2,
        polewidth / 2,
        polelen - polewidth / 2,
        -polewidth / 2,
    )
    pole.v = [(l, b), (l, t), (r, t), (r, b)]

    x = state
    cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
    carttrans.set_translation(cartx, carty)
    poletrans.set_rotation(-x[2])

    return viewer.render(return_rgb_array=mode == "rgb_array"), {
        "viewer": viewer,
        "pole": pole,
        "carttrans": carttrans,
        "poletrans": poletrans,
    }
