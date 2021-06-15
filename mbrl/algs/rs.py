"""
Implement Random Shooting core functions.
"""

import tax
import jax
import chex
import jax.numpy as jnp

from jax import jit
from jax import partial
from typing import Optional
from typing import Callable


@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def trajectory_search(rng, horizon, action_dim, minval = None, maxval = None, action_type: str = 'continuous') -> jnp.DeviceArray:
    """ Generate Sequence of action of length `horizon`"""
    if action_type == 'continuous':
        x = jax.random.uniform(
            rng, (horizon, action_dim),
            minval=minval, maxval=maxval,
        )
    if action_type == 'discrete':
        # We considere the action set: {0, .., maxval}
        chex.assert_type(action_dim, int)
        x = jax.random.choice(rng, action_dim, shape=(horizon,))
    return x


@partial(jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def forecast(rng, ob_0, step_fn, horizon, action_dim, minval, maxval, action_type: str = 'continuous'):
    """ Generate a trajectory by iteratively leverage a world model (`step_fn`) """
    rng = jax.random.split(rng, 1 + horizon)
    traj = trajectory_search(rng[0], horizon, action_dim, minval, maxval, action_type)
    init = (rng[1:], ob_0, traj)
    time = jnp.arange(horizon)
    _, out = jax.lax.scan(step_fn, init, time)
    return out


@partial(jit, static_argnums=(1, 2))
def score(history: dict, discount: float = 0.99,
          terminal_reward_fn: Optional[Callable] = None):
    """Score a truncated trace of episode (history)."""
    rewards = history["reward"]
    if terminal_reward_fn is not None:
        last_reward = terminal_reward_fn(history['observation_next'][-1])
        rewards = jnp.concatenate([rewards, last_reward])
    cum_rewards = tax.discounted_return(rewards, discount)
    info = {**history, "score": cum_rewards[0], "cum_rewards": cum_rewards}
    return info['score'], info


@partial(jit, static_argnums=(2, 3, 4,))
def plan(rng: chex.Array, ob_0: chex.Array,
         forecast: Callable, score: Callable,
         population: int = 2000):
    rng = jax.random.split(rng, population)
    out = jax.vmap(forecast, (0, None))(rng, ob_0)
    value, info = jax.vmap(score, 0)(out)
    index = jnp.argmax(value)
    return info['action'][index], info
