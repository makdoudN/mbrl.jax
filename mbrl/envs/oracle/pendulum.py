import os
import jax
import numpy as np

import jax.numpy as jnp
from jax import jit
from gym.envs.classic_control import rendering


# Default environment parameters for Pendulum-v0
env_params = {
    "max_speed": 8,
    "max_torque": 2.0,
    "dt": 0.05,
    "g": 10.0,
    "m": 1.0,
    "l": 1.0,
}


def step_pendulum(params, state, u):
    """ Integrate pendulum ODE and return transition. """
    th, thdot = state[0], state[1]
    u = jnp.clip(u, -params["max_torque"], params["max_torque"])
    costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

    newthdot = (
        thdot
        + (
            -3 * params["g"] / (2 * params["l"]) * jnp.sin(th + jnp.pi)
            + 3.0 / (params["m"] * params["l"] ** 2) * u
        )
        * params["dt"]
    )
    newth = th + newthdot * params["dt"]
    newthdot = jnp.clip(newthdot, -params["max_speed"], params["max_speed"])

    state = jnp.array([newth, newthdot])
    return state.squeeze(-1), get_obs_pendulum(state), -costs[0].squeeze(), False, {}


def reset_pendulum(rng):
    """ Reset environment state by sampling theta, thetadot. """
    high = jnp.array([jnp.pi, 1])
    state = jax.random.uniform(rng, shape=(2,), minval=-high, maxval=high)
    return state, get_obs_pendulum(state)


def get_obs_pendulum(state):
    """ Return angle in polar coordinates and change. """
    th, thdot = state[0], state[1]
    return jnp.array([jnp.cos(th), jnp.sin(th), thdot]).squeeze()


def angle_normalize(x):
    """ Normalize the angle - radians. """
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


reset = jit(reset_pendulum)
step = jit(step_pendulum)


def render():
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
    dirname = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(dirname, "./assets/clockwise.png")
    img = rendering.Image(fname, 1.0, 1.0)
    imgtrans = rendering.Transform()
    img.add_attr(imgtrans)
    state_env, a_tm1 = yield
    while True:
        viewer.add_onetime(img)
        pole_transform.set_rotation(state_env[0] + np.pi / 2)
        imgtrans.scale = (-a_tm1 / 2, np.abs(a_tm1) / 2)
        viewer.render()
        state_env, a_tm1 = yield
