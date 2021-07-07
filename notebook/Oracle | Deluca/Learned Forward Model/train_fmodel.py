"""
Training Forward Model Utilities.
"""


import jax
import tax
import tqdm
import collections
import haiku as hk
import numpy as np
import jax.numpy as jnp
import typing
import optax
import chex
import tree
import haiku as hk
from jax import jit
from jax import vmap
from jax import partial
from mbrl.common.nn import mlp_deterministic
from mbrl.common.nn import mlp_multivariate_normal_diag
from torch.utils.data import DataLoader

@partial(jit, static_argnums=(3, 4))
def update_fn(state, inputs, target, opt, loss_fn):
    l, g = jax.value_and_grad(loss_fn)(state.params, inputs, target)
    updates, opt_state = opt.update(g, state.opt_state)
    params = jax.tree_multimap(lambda p, u: p + u, state.params, updates)
    state = state.replace(params=params, opt_state=opt_state)
    metrics = {"loss": l}
    return state, metrics


@chex.dataclass
class NormalizationState:
    observation_mean: jnp.ndarray
    observation_std: jnp.ndarray
    action_mean: jnp.ndarray
    action_std: jnp.ndarray


@chex.dataclass
class FState:
    params: typing.Any
    opt_state: typing.Any
    norm: NormalizationState


def init_fmodel_training(
    rng,
    observation_size,
    action_size,
    fmodel_type: str,  # [D, P]
    fmodel_kwargs: dict,
    optim: str,
    optim_kwargs: dict,
    fmodel_training_kwargs: dict,
    type_loss: str = "l1"
    # TODO: Add ensemble possibilities
):
    # NOTE.1. `type_loss` is ignore if fmodel_type is not 'D'.
    dummy_obs = jnp.zeros((observation_size,))
    dummy_acs = jnp.zeros((action_size,))

    state_norm = NormalizationState(
        observation_mean=jnp.zeros((observation_size,)),
        observation_std=jnp.ones((observation_size,)),
        action_mean=jnp.zeros((action_size,)),
        action_std=jnp.ones((action_size,)),
    )

    # -- Setup NN approximation of the forward model.
    fmodel_opt = getattr(optax, optim)(learning_rate=1e-3)
    fmodel_cls = (
        mlp_deterministic if fmodel_type == "D" else mlp_multivariate_normal_diag
    )

    fmodel_def = lambda x, a: fmodel_cls(observation_size, **fmodel_kwargs)(
        jnp.concatenate([x, a], -1)
    )
    fmodel_def = hk.transform(fmodel_def)
    fmodel_def = hk.without_apply_rng(fmodel_def)
    rng, rng_params = jax.random.split(rng)
    fmodel_params = fmodel_def.init(rng_params, dummy_obs, dummy_acs)
    fmodel_opt_state = fmodel_opt.init(fmodel_params)

    # -- Build the forward state
    fstate = FState(params=fmodel_params, opt_state=fmodel_opt_state, norm=state_norm)

    # -- Build the loss function associated with an update function.
    if fmodel_type == "D":
        @partial(jit, static_argnums=(3, 4))
        def loss_fn(p, inputs, target, fmodel_def, type_loss):
            prediction = fmodel_def(p, *inputs)
            if type_loss == "l1":
                loss = tax.l1_loss(prediction, target)
            else:
                loss = tax.l2_loss(prediction, target)
            loss = loss.mean()
            return loss

        loss = jit(partial(loss_fn, fmodel_def=fmodel_def.apply, type_loss=type_loss))

    if fmodel_type == "P":
        @partial(jit, static_argnums=(3,))
        def loss_fn(p, inputs, target, fmodel_def):
            dist = fmodel_def(p, *inputs)
            loss = -dist.log_prob(target).mean()
            return loss

        loss = jit(partial(loss_fn, fmodel_def=fmodel_def.apply))

    update = jit(partial(update_fn, loss_fn=loss, opt=fmodel_opt))

    train_step = partial(
        train_fmodel,
        loss_fn=loss,
        update_fn=update,
        forward_model=jit(fmodel_def.apply),
        **fmodel_training_kwargs)

    return fstate, train_step


# Should return update_step: fn


def train_fmodel(
    data,
    state,
    loss_fn,
    update_fn,
    forward_model,
    seed: int = 42,
    batch_size: int = 256,
    use_norm: bool = True,
    use_residual: bool = True,
    max_epochs: int = 200,
    validation_size: float = 0.25,
    early_stopping_patience: int = 15,
    alpha_norm: float = 0.5,
):
    data = tree.map_structure(lambda v: np.array(v), data)

    # Assume that the initial normalization is (mean=0, std=1)
    # This setting correspond to no normalization.
    # If the flag `use_norm` is True, the function
    # will update the norm state based on the data
    # it received according a polyak rule (weighted update).
    if use_norm:
        observation_mean = data["observation"].mean(0)
        observation_std = data["observation"].std(0)
        action_mean = data["action"].mean(0)
        action_std = data["action"].std(0)
        new_observation_mean = (
            1 - alpha_norm
        ) * observation_mean + alpha_norm * state.norm.observation_mean
        new_observation_std = (
            1 - alpha_norm
        ) * observation_std + alpha_norm * state.norm.observation_std
        new_action_mean = (
            1 - alpha_norm
        ) * action_mean + alpha_norm * state.norm.action_mean
        new_action_std = (
            1 - alpha_norm
        ) * action_std + alpha_norm * state.norm.action_std

        new_norm = NormalizationState(
            observation_mean=new_observation_mean,
            observation_std=new_observation_std,
            action_mean=new_action_mean,
            action_std=new_action_std,
        )
        state = state.replace(norm=new_norm)

    def process(observation, action, observation_next):
        observation_norm = (observation - state.norm.observation_mean) / (
            state.norm.observation_std + 1e-6
        )
        action_norm = (action - state.norm.action_mean) / (state.norm.action_std + 1e-6)
        inputs = (observation_norm, action_norm)
        target = observation_next
        if use_residual:
            target = observation_next - observation
        return inputs, target

    ds = tax.DatasetDict(data)
    es = tax.EarlyStopping(patience=early_stopping_patience)
    ds_train, ds_valid = tax.random_splits(ds, validation_size)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = DataLoader(ds_valid, batch_size=batch_size)

    allinfo = []
    t = tqdm.trange(max_epochs)
    for range in t:
        store = tax.Store(decimals=4)
        for batch in dl_train:
            batch = tree.map_structure(lambda v: jnp.asarray(v), batch)
            action = batch["action"]
            observation = batch["observation"]
            observation_next = batch["observation_next"]
            inputs, target = process(observation, action, observation_next)
            state, info = update_fn(state, inputs, target)
            store.add(**{"loss/train": info["loss"]})
        for batch in dl_valid:
            batch = tree.map_structure(lambda v: jnp.asarray(v), batch)
            action = batch["action"]
            observation = batch["observation"]
            observation_next = batch["observation_next"]
            inputs, target = process(observation, action, observation_next)
            loss_val = loss_fn(state.params, inputs, target)
            store.add(**{"loss/valid": loss_val})

        metrics = store.get()
        t.set_postfix(metrics)
        allinfo.append(metrics)
        validation_loss = metrics["loss/valid"]
        if es.step(validation_loss):
            break

    # Sanity Check to automatically choose the right
    # inference model.
    dummy_prediction = forward_model(state.params, observation, action)

    model = "P"
    if isinstance(dummy_prediction, jnp.ndarray):
        model = "D"

    """
    # At this point, the training is done.
    # We return the state, the metrics
    # and the forward model inference ready
    # to use according the training.
    @jit
    def fmodel_inference(rng, observation, action):
        observation_norm = (observation - state.norm.observation_mean) / (
            state.norm.observation_std + 1e-6
        )
        action_norm = (action - state.norm.action_mean) / (state.norm.action_std + 1e-6)

        prediction = forward_model(state.params, observation_norm, action_norm)
        if model == "P":
            prediction = prediction.loc       # Test prediction.loc.val
        if use_residual:
            return prediction + observation
        return prediction
    """
    #params = hk.data_structures.to_immutable_dict(state.params)
    fmodel_inference_ = partial(fmodel_inference, 
                                forward_def=forward_model,
                                params=state.params, 
                                norm_state=state.norm,
                                model=model,
                                use_residual=use_residual)
    
    info = tax.reduce(allinfo)
    return state, jit(fmodel_inference_), info




@partial(jit, static_argnums=(3, 6, 7))
def fmodel_inference(rng, observation, action, forward_def, params, norm_state, model, use_residual):
    observation_norm = (observation - norm_state.observation_mean) / (
        norm_state.observation_std + 1e-6
    )
    action_norm = (action - norm_state.action_mean) / (norm_state.action_std + 1e-6)

    prediction = forward_def(params, observation_norm, action_norm)
    if model == "P":
        prediction = prediction.loc       # Test prediction.loc.val
    if use_residual:
        return prediction + observation
    return prediction