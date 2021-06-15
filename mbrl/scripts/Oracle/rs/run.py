import gym
import tax
import jax
import yaml
import hydra
import tqdm
import importlib
import collections
import jax.numpy as jnp

import mbrl
import mbrl.envs

from jax import jit
from jax import partial
from omegaconf import OmegaConf
from mlrec.recorder import Recorder
from mbrl.algs.rs import score as score_rs
from mbrl.algs.rs import plan as plan_rs
from mbrl.algs.rs import forecast


tax.set_platform("cpu")
Environment = collections.namedtuple("Environment", ["step", "reset"])


@partial(jit, static_argnums=(2,))
def world_oracle(carry, t, env):
    keys, (state, observation), trajectory = carry
    action = trajectory[t]
    state_next, observation_next, reward, done, info = env.step(state, action)
    reward = reward.astype(jnp.float32)
    carry = keys, (state_next, observation_next), trajectory
    return carry, {
        "observation": observation,
        "observation_next": observation_next,
        "state": state,
        "state_next": state_next,
        "reward": reward,
        "action": action,
        "terminal": 1.0 - done,
    }


@partial(jit, static_argnums=(2, 3))
def step_interaction(carry, t, env, policy):
    key, (env_state, observation) = carry
    key, subkey = jax.random.split(key)
    action, _ = policy(subkey, env_state, observation)
    env_state_next, observation_next, reward, terminal, info = env.step(
        env_state, action
    )
    carry = key, (env_state_next, observation_next)
    return carry, {
        "observation": observation,
        "observation_next": observation_next,
        "reward": reward,
        "action": action,
        "terminal": 1 - terminal,
        "env_state": env_state,
        "env_state_next": env_state_next,
    }


def init_policy(
    conf, world, action_type, action_size, action_max=None, action_min=None
):
    score_fn = partial(score_rs, discount=conf.discount)
    score_fn = jit(score_rs)
    forecast_fn = partial(
        forecast,
        step_fn=world,
        horizon=conf.horizon,
        action_dim=action_size,
        minval=action_min,
        maxval=action_max,
        action_type=action_type,
    )
    forecast_fn = jit(forecast_fn)
    plan = partial(
        plan_rs, score=score_fn, forecast=forecast_fn, population=conf.population
    )
    plan = jit(plan)

    @jit
    def policy(rng, s, x):
        trjs, info = plan(rng, (s, x))
        action = trjs[0]
        return action, info

    return policy


@hydra.main(config_path=".", config_name="conf")
def main(conf):
    print(conf)
    rng = jax.random.PRNGKey(conf.seed)
    rec = Recorder(output_dir=".")
    yconf = OmegaConf.to_yaml(conf, resolve=True)
    print(yconf)
    rec.save(yaml.safe_load(yconf), "conf.yaml")

    # -- Initialize the Environment.
    envlib = importlib.import_module(conf.env_root)
    envpar = getattr(envlib, "env_params")
    step = getattr(envlib, "step_fn")
    step = partial(step, env_params=envpar)
    reset = getattr(envlib, "reset_fn")
    reset = partial(reset, env_params=envpar)
    reward = getattr(envlib, "reward_fn")
    reward = partial(reward, env_params=envpar)
    env = Environment(step=jit(step), reset=jit(reset))
    max_episode_steps = envpar["max_episode_steps"]
    action_size = envpar["action_size"]
    action_min = envpar.get("action_min", None)
    action_max = envpar.get("action_max", None)
    action_type = envpar.get("action_type", "continuous")
    world = partial(world_oracle, env=env)

    for e in tqdm.trange(conf.epochs):
        rng, rng_reset = jax.random.split(rng, 2)
        policy = init_policy(
            conf,
            world,
            action_type=action_type,
            action_size=action_size,
            action_min=action_min,
            action_max=action_max,
        )
        env_state, observation = env.reset(rng_reset)
        init = (rng, (env_state, observation))
        step = partial(step_interaction, policy=policy, env=env)
        _, out = jax.lax.scan(step, init, jnp.arange(max_episode_steps))

        rec.write(
            {
                "epoch": e,
                "score": sum(out["reward"]),
            }
        )

    print("End")


if __name__ == "__main__":
    main()
