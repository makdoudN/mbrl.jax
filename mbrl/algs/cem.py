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
from mbrl.algs.rs import score


@partial(jit, static_argnums=(3, 4, 5, 6))
def trajectory_search(rng, loc, scale, horizon, action_dim, minval, maxval) -> jnp.DeviceArray:
    """ Generate Sequence of action of length `horizon`"""
    shape = (horizon, action_dim)
    x = loc + jax.random.normal(rng, shape) * scale
    x = jnp.clip(x, minval, maxval)
    return x


@partial(jit, static_argnums=(4, 5, 6, 7, 8))
def forecast(rng, ob_0, loc, scale, step_fn,
             horizon, action_dim, minval, maxval):
    """ Generate a trajectory by iteratively leverage a world model (`step_fn`) """
    rng = jax.random.split(rng, 1 + horizon)
    traj = trajectory_search(rng[0], loc, scale, horizon, action_dim, minval, maxval)
    init = (rng[1:], ob_0, traj)
    time = jnp.arange(horizon)
    _, out = jax.lax.scan(step_fn, init, time)
    return out


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


@partial(jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
def plan(rng: chex.Array, ob_0: chex.Array,
         forecast: Callable, score: Callable,
         action_dim: int,
         population: int = 500, alpha: float = 0.25,
         nelites: int = 50, niters: int = 4,
         horizon: int = 20,
         stddev_init: float = 3):

    vmap_score = jit(vmap(score, 0))
    vmap_forecast_fn = jit(vmap(forecast, (0, None, None, None)))

    @jit
    def one_step_cem(carry, xs):
        rng, ob_0, loc, scale = carry
        rng, key = jax.random.split(rng)
        key = jax.random.split(key, population)
        out = vmap_forecast_fn(key, ob_0, loc, scale)
        _, out = vmap_score(out)
        (elite_loc, elite_scale), out = get_elite_stats(out, nelites=nelites)
        chex.assert_equal_shape([loc, elite_loc])
        chex.assert_equal_shape([scale, elite_scale])
        loc = alpha * loc + (1 - alpha) * elite_loc
        scale = alpha * scale + (1 - alpha) * elite_scale
        out = {**out, "loc_new": loc, "scale_new": scale}
        return (rng, ob_0, loc, scale), out

    loc = jnp.zeros((horizon, action_dim))
    scale = jnp.ones((horizon, action_dim)) * stddev_init
    init = (rng, ob_0, loc, scale)
    _, out = jax.lax.scan(one_step_cem, init, jnp.arange(niters))
    actions = out["loc_new"][-1]
    return actions, out
