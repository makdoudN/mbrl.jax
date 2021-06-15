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
"""
@author: Olivier Sigaud
A merge between two sources:
* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia
* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
# NOTE: State is Observation
import math

import jax
import jax.numpy as jnp
import numpy as np


env_params = dict(
    min_action=-1.0,
    max_action=1.0,
    min_position=-1.2,
    max_position=0.6,
    max_speed=0.07,
    goal_position=0.45,  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
    goal_velocity=0.0,
    power=0.0015,
    H=50,
    action_dim=1,
    action_size=1,
)
# Comply with API
env_params['max_episode_steps'] = env_params['H']
env_params['action_min'] = env_params['min_action']
env_params['action_max'] = env_params['max_action']


def dynamics(state, action, env_params):
    position = state[0]
    velocity = state[1]
    force = jnp.minimum(
        jnp.maximum(action, env_params["min_action"]), env_params["max_action"]
    )
    velocity += force * env_params["power"] - 0.0025 * jnp.cos(3 * position)
    velocity = jnp.clip(velocity, -env_params["max_speed"], env_params["max_speed"])
    position += velocity
    position = jnp.clip(
        position, env_params["min_position"], env_params["max_position"]
    )
    reset_velocity = (position == env_params["min_position"]) & (velocity < 0)
    velocity = jax.lax.cond(
        reset_velocity[0], lambda x: jnp.zeros((1,)), lambda x: x, velocity
    )
    return jnp.concatenate([position, velocity], -1)


def terminal_fn(state, env_params):
    position = state[0]
    velocity = state[1]
    condition = (position >= env_params["goal_position"]) & (
        velocity >= env_params["goal_velocity"]
    )
    done = jax.lax.cond(condition, (), lambda _: 1.0, (), lambda _: 0.0)
    return done


def cost(state, u, env_params):
    done = terminal_fn(state, env_params)
    return -100.0 * done + 0.1 * (u[0] + 1) ** 2


def reward_fn(state, u, env_params):
    return -cost(state, u, env_params)


def step_fn(state, action, env_params):
    state = dynamics(state, action, env_params)
    done = terminal_fn(state, env_params)
    reward = reward_fn(state, action, env_params)
    return state, state, reward, done, {}


def reset_fn(rng, env_params=None):
    state = jnp.array([jax.random.uniform(rng, minval=-0.6, maxval=0.4), 0])
    return state, state


def _height(xs):
    return jnp.sin(3 * xs) * 0.45 + 0.55


def render(state, kwargs, mode="human", env_params=env_params):
    screen_width = 600
    screen_height = 400
    viewer = kwargs.get("viewer", None)
    cartrans = kwargs.get("cartrans", None)
    max_position = env_params['max_position']
    min_position = env_params['min_position']
    goal_position = env_params['goal_position']
    world_width = max_position - min_position
    scale = screen_width / world_width
    carwidth = 40
    carheight = 20

    if viewer is None:
        from gym.envs.classic_control import rendering

        viewer = rendering.Viewer(screen_width, screen_height)
        xs = np.linspace(min_position, max_position, 100)
        ys = _height(xs)
        xys = list(zip((xs - min_position) * scale, ys * scale))
        track = rendering.make_polyline(xys)
        track.set_linewidth(4)
        viewer.add_geom(track)
        clearance = 10
        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        car.add_attr(rendering.Transform(translation=(0, clearance)))
        cartrans = rendering.Transform()
        car.add_attr(cartrans)
        viewer.add_geom(car)
        frontwheel = rendering.make_circle(carheight / 2.5)
        frontwheel.set_color(0.5, 0.5, 0.5)
        frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
        frontwheel.add_attr(cartrans)
        viewer.add_geom(frontwheel)
        backwheel = rendering.make_circle(carheight / 2.5)
        backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
        backwheel.add_attr(cartrans)
        backwheel.set_color(0.5, 0.5, 0.5)
        viewer.add_geom(backwheel)
        flagx = (goal_position - min_position) * scale
        flagy1 = _height(goal_position) * scale
        flagy2 = flagy1 + 50
        flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
        viewer.add_geom(flagpole)
        flag = rendering.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )
        flag.set_color(0.8, 0.8, 0)
        viewer.add_geom(flag)
        kwargs['viewer'] = viewer
        kwargs['cartrans'] = cartrans

    pos = state[0]
    cartrans.set_translation(
        (pos - min_position) * scale, _height(pos) * scale
    )
    cartrans.set_rotation(math.cos(3 * pos))

    return viewer.render(return_rgb_array=mode == "rgb_array"), kwargs
