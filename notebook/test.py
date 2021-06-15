import jax
import tax
import clu
import time
import tqdm
import tree
import haiku as hk
import numpy as np
import collections
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import mbrl
import brax
import tqdm
import functools

from jax import jit
from brax import envs
from brax.io import html
from functools import partial
from mbrl.algs.rs import trajectory_search, forecast, score, plan


tax.set_platform('gpu')


def train():
    rng = jax.random.PRNGKey(42)

    name = 'halfcheetah'
    envf = envs.create_fn(name)
    env = envf()
    action_size = env.action_size

    @jit
    def step(carry, t):
        rng, env_state, action_trajectory = carry
        action = action_trajectory[t]
        env_state_next = env.step(env_state, action)
        carry = (rng, env_state_next, action_trajectory)

        info = dict(
            observation=env_state.obs,
            observation_next=env_state.obs,
            reward=env_state_next.reward,
            terminal=1 - env_state_next.done,
            action=action,
            env_state=env_state,
            env_state_next=env_state_next,
        )
        return carry, info

    forecast_ = partial(
        forecast, step_fn=step,
        horizon=20, action_dim=action_size,
        minval=-1, maxval=1,
    )

    @jit
    def one_step_interaction(carry, t):
        rng, env_state = carry
        action = plan(rng, env_state, forecast_, score)[0][0]
        env_state_next = env.step(env_state, action)
        carry = (rng, env_state_next)

        info = dict(
            observation=env_state.obs,
            observation_next=env_state.obs,
            reward=env_state_next.reward,
            terminal=1 - env_state_next.done,
            action=action,
        )
        return carry, info

    #
    # Training
    #

    start = time.time()
    env_state = env.reset(rng)
    init = (rng, env_state)
    _, out = jax.lax.scan(one_step_interaction, init, jnp.arange(1000))  # First should be long.

    print('Data -> Replay')
    rb = tax.ReplayBuffer(100_000)
    rb.add(**{
        'observation': out['observation'],
        'observation_next': out['observation_next'],
        'reward': out['reward'],
        'action': out['action'],
        'terminal': out['terminal']
    })
    print(len(rb))
    print(rb.sample(5))

    end = time.time()
    print(f"Elapsed Time: {end - start}")

    start = time.time()
    env_state = env.reset(rng)
    init = (rng, env_state)
    _, out = jax.lax.scan(one_step_interaction, init, jnp.arange(1000))  # First should be long.

    print('Data -> Replay')
    rb = tax.ReplayBuffer(100_000)
    rb.add(**{
        'observation': out['observation'],
        'observation_next': out['observation_next'],
        'reward': out['reward'],
        'action': out['action'],
        'terminal': out['terminal']
    })
    print(len(rb))
    print(rb.sample(5))

    end = time.time()
    print(f"Elapsed Time: {end - start}")


if __name__ == "__main__":
    train()


