"""
Implement Cross Entropy Methods core functions.
"""

import tax
import jax
import chex
import jax.numpy as jnp
import collections

from jax import jit
from jax import vmap
from jax import partial
from typing import Optional
from typing import Callable


@partial(jit, static_argnums=(3, 4, 5, 6))
def trajectory_search(rng, loc, scale, horizon, action_dim, minval, maxval) -> jnp.DeviceArray:
    """ Generate Sequence of action of length `horizon`"""
    shape = (horizon, action_dim)
    x = loc + jax.random.normal(rng, shape) * scale
    x = jnp.clip(x, minval, maxval)
    return x


@partial(jit, static_argnums=(4, 5, 6, 7, 8))
def forecast(rng, ob_0, loc, scale, step_fn, horizon, action_dim, minval, maxval):
    """ Generate a trajectory by iteratively leverage a world model (`step_fn`) """
    rng = jax.random.split(rng, 1 + horizon)
    traj = trajectory_search(rng[0], loc, scale, horizon, action_dim, minval, maxval)
    init = (rng[1:], ob_0, traj)
    time = jnp.arange(horizon)
    _, out = jax.lax.scan(step_fn, init, time)
    return out


@partial(jit, static_argnums=(1,))
def score(history: dict, discount: float = 0.99,
          terminal_reward_fn: Optional[Callable] = None):
    """Score a truncated trace of episode (history)."""
    rewards = history["reward"]
    if terminal_reward_fn is not None:
        last_reward = terminal_reward_fn(history['observation_next'][-1])
        rewards = jnp.concatenate([rewards, last_rewards])
    cum_rewards = tax.discounted_return(rewards, discount)
    info = {**history, "score": cum_rewards[0], "cum_rewards": cum_rewards}
    return info['score'], info


@jax.partial(jit, static_argnums=(1, 2, 3))
def get_elite_stats(batched_history, 
                    nelites: int = 50,
                    score_index: str = 'score', 
                    action_index: str = 'action'):
    # -- Extract the index of the K best rollout
    v_t = batched_history[score_index]
    a_t = batched_history[action_index]
    index = jnp.argsort(v_t)[::-1]              # sorted index by value.
    index_elite = index[:nelites]               # take the K best one. 

    # -- Compute the statistic of the elites
    elite_action = a_t[index_elite]
    elite_loc = jnp.mean(elite_action, 0)
    elite_scale = jnp.std(elite_action, 0)
    
    batched_history = {
        **batched_history,
        "elite_loc": elite_loc,
        "elite_scale": elite_scale,
        "index_elite": index_elite,
        "index_sorted_by_score": index,
    }
    return (elite_loc, elite_scale), batched_history