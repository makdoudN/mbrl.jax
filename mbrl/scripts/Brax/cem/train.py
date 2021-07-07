# Training Scripts of Random Shooting for Brax Environment.

import tax
import jax
import yaml
import tqdm
import hydra
import functools
import jax.numpy as jnp
import mbrl
import brax

from jax import jit
from brax import envs
from omegaconf import OmegaConf
from squirrel.recorder import Recorder
from mbrl.algs.rs import forecast, plan
from mbrl.algs.rs import score as score_trajectory


tax.set_platform("cpu")


def save_conf(rec, conf, display: bool = True):
    yamlconf = OmegaConf.to_yaml(conf, resolve=True)
    if display:
        print(yamlconf)
    yamlconf = yaml.safe_load(yamlconf)
    rec.save(yamlconf, 'conf.yaml')


@hydra.main(config_path="conf", config_name="cem")
def main(conf):
    rng = jax.random.PRNGKey(conf.seed)
    rec = Recorder(output_dir=".")
    save_conf(rec, conf)

    # -- initialization Environment.

    env_fn = envs.create_fn(conf.env)
    env = env_fn()
    action_size = env.action_size

    @jit
    def step_rs(carry, t):
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

    # -- Setup Random Shooting function.

    forecast_ = functools.partial(
        forecast, step_fn=step_rs,
        horizon=conf.rs.horizon, action_dim=action_size,
        minval=-1, maxval=1,
    )
    score = functools.partial(score_trajectory, discount=conf.discount, terminal_reward_fn=None)

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
            env_state=env_state,
            env_state_next=env_state_next,
        )
        return carry, info

    for i in tqdm.trange(conf.niters):
        rng, subrng = jax.random.split(rng)
        env_state = env.reset(subrng)
        init = (subrng, env_state)
        _, info = jax.lax.scan(one_step_interaction, init, jnp.arange(1000))
        score = float(info['reward'].sum())
        metrics = {
            'score': score,
            'epoch': i
        }
        rec.write(metrics)



if __name__ == "__main__":
    main()
