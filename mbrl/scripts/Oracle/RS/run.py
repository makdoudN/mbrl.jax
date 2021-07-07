import jax
import tax
import numpy as np
import collections
import jax.numpy as jnp
import importlib
import tree
import mbrl
import yaml
import hydra
from jax import jit
from jax import partial
from omegaconf import OmegaConf
from mlrec.recorder import Recorder
from mbrl.algs.rs import score as score_rs
from mbrl.algs.rs import plan as plan_rs
from mbrl.algs.rs import forecast
from utils import init_fmodel_training

tax.set_platform("cpu")
Environment = collections.namedtuple("Environment", ["step", "reset"])


@hydra.main(config_path="conf", config_name="base")
def main(conf):
    rng = jax.random.PRNGKey(conf.seed)
    rec = Recorder(output_dir=".")
    yconf = OmegaConf.to_yaml(conf, resolve=True)
    rec.save(yaml.safe_load(yconf), "conf.yaml")
    print(yconf)

    # -- Initialize the Environment.
    envlib = importlib.import_module(conf.env_root)
    envpar = getattr(envlib, "env_params")
    step = getattr(envlib, "step_fn")
    step = partial(step, env_params=envpar)
    reset = getattr(envlib, "reset_fn")
    reset = partial(reset, env_params=envpar)
    reward_fn = getattr(envlib, "reward_fn")
    reward_fn = partial(reward_fn, env_params=envpar)
    env = Environment(step=jit(step), reset=jit(reset))
    action_min = envpar.get("action_min", None)
    action_max = envpar.get("action_max", None)
    action_size = envpar["action_size"]
    action_type = envpar.get("action_type", "continuous")
    max_episode_steps = envpar["max_episode_steps"]

    rng, rng_init = jax.random.split(rng)
    observation_size = len(env.reset(rng_init)[-1])

    # -- Replay Buffer Init
    rb = tax.ReplayBuffer(100_000)

    rng, subrng = jax.random.split(rng)
    state, fit = init_fmodel_training(
        rng, observation_size, action_size,
        action_type=action_type,
        **conf.fmodel
    )

    # Initialize of the Algorithm
    # 1. Gather random interaction with the environment
    # 2. Initiale the model.
    while len(rb) < 2000:
        score, buf = 0, []
        env_state, observation = env.reset(rng)
        for _ in range(max_episode_steps):
            rng, key = jax.random.split(rng)
            if action_type == 'continuous':
                action = jax.random.uniform(key, (1,), minval=action_min, maxval=action_max)
            else:
                action = jax.random.choice(key, action_size)
            env_state_next, observation_next, reward, terminal, info = env.step(env_state, action)
            score += reward
            buf.append({
                'observation': observation,
                'observation_next': observation_next,
                'action': action,
            })
            observation = observation_next.copy()
            env_state = env_state_next
        data = tax.reduce(buf, np.stack)
        rb.add(**data)

    print(f'Replay Buffer size is {len(rb)}')

    state, fmodel_inference, info = fit(rb.dataset(), state)

    #
    # Training Loop
    #

    for e in range(conf.nepochs):
        store = tax.Store()

        # NOTE: Terminal=False is not possible
        # --> partial world with reward and terminal function
        # --> if not provided set terminal to false.
        # NOTE: GT Model to compare and anylyse the planning model.
        # NOTE (%) = The state is ignore.
        # NOTE = Reward function [observation_next/observation]
        # NOTE = Add dynamics to compare
        @jit
        def world(carry, t):
            keys, (state, observation), trajectory = carry
            key = keys[t]
            action = trajectory[t]
            # -- Forward Model
            observation_next = fmodel_inference(key, observation, action)
            reward = reward_fn(observation, action)
            carry = keys, (state, observation_next), trajectory
            return carry, {
                'key': key,
                "observation": observation,
                "observation_next": observation_next,
                "reward": reward,
                "action": action,
            }

        score_ = partial(score_rs, terminal_reward_fn=None, discount=0.99)
        forecast_ = partial(
            forecast,
            step_fn=world,
            horizon=conf.horizon,
            action_type=action_type,
            action_dim=action_size,
            minval=action_min,
            maxval=action_max
        )

        policy = partial(plan_rs, forecast=forecast_, score=score_, population=conf.population)
        policy = jit(policy)

        @jit
        def one_step(carry, t):
            """ Entire Loop with scan"""
            key, (env_state, observation)  = carry
            key, subkey = jax.random.split(key)
            action, action_info = policy(subkey, (env_state, observation))
            action = action[0]
            env_state_next, observation_next, reward, terminal, info = \
                env.step(env_state, action)
            carry = key, (env_state_next, observation_next)
            return carry, {
                "observation": observation,
                "observation_next": observation_next,
                "reward": reward,
                "action": action,
                "terminal": 1 - terminal,
                "env_state": env_state,
                'env_state_next': env_state_next,
                "action_info": action_info,
            }

        # -- Interaction with the environment
        for _ in range(conf.niters):
            rng, subrng = jax.random.split(rng)
            env_state, observation = env.reset(subrng)
            init = (rng, (env_state, observation))
            _, out = jax.lax.scan(one_step, init, jnp.arange(max_episode_steps))
            score = jnp.sum(out['reward'])
            store.add(score=score)

            action = out['action']
            env_state = out['env_state']
            env_state_next = out['env_state_next']
            rb.add(action=action,
                   env_state=env_state,
                   env_state_next=env_state_next)

        # -- Training of the model
        state, fmodel_inference, info_training = fit(rb.dataset(), state)
        info_training = tree.map_structure(lambda v: float(v[-1]), info_training)

        # -- Log
        info = store.get()
        info.update({'epoch': e})
        info.update(info_training)
        rec.write(info)


if __name__ == "__main__":
    main()
