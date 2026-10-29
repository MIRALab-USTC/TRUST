from dataclasses import dataclass
from typing import Any
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
from vmax.agents import datatypes
from vmax.agents.networks.distributions import NormalTanhDistribution
from .losses import (
    awr_loss,
    critic_targets,
    expectile_loss,
    iqr_weights,
    per_transition_critic_loss,
    squashed_gaussian_log_prob,
)


@flax.struct.dataclass
class TrustNetworkParams:
    policy: Any
    value: Any
    target_value: Any
    state_value: Any


@flax.struct.dataclass
class TrustTrainingState:
    params: TrustNetworkParams
    policy_optimizer_state: Any
    value_optimizer_state: Any
    state_value_optimizer_state: Any
    rl_gradient_steps: Any = 0


class TrustMLP(nn.Module):
    layer_sizes: tuple[int, ...]
    output_size: int = 1

    @nn.compact
    def __call__(self, obs):
        x = obs
        for i, size in enumerate(self.layer_sizes):
            x = nn.relu(nn.Dense(size, name=f"hidden_{i}")(x))
        return nn.Dense(self.output_size, name="output")(x)


class TrustCritic(nn.Module):
    layer_sizes: tuple[int, ...]
    num_networks: int = 1

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        return jnp.concatenate(
            [TrustMLP(self.layer_sizes, name=f"critic_{i}")(x) for i in range(self.num_networks)],
            axis=-1,
        )


@dataclass(frozen=True)
class Network:
    init: Any
    apply: Any


@dataclass(frozen=True)
class TrustNetworks:
    policy_network: Network
    value_network: Network
    state_value_network: Network
    parametric_action_distribution: Any
    policy_optimizer: Any
    value_optimizer: Any
    state_value_optimizer: Any


def make_networks(
    observation_size: int,
    action_size: int,
    unflatten_fn=None,
    learning_rate: float = 0.0001,
    network_config: dict | None = None,
) -> TrustNetworks:
    config = network_config or {}
    if config.get("encoder", {}).get("type", "none") != "none":
        raise ValueError("The paper TRUST configuration uses direct vector MLPs (encoder=none).")
    if config.get("action_distribution", "gaussian") != "gaussian":
        raise ValueError("TRUST uses a tanh-squashed Gaussian action distribution.")

    def layers(name):
        block = config.get(name, {})
        if block.get("activation", "relu") != "relu" or block.get("final_activation") is not None:
            raise ValueError("TRUST uses ReLU hidden layers and linear output layers.")
        result = tuple(block.get("layer_sizes", [256, 64, 32]))
        if not result or any((size <= 0 for size in result)):
            raise ValueError("Network hidden layer sizes must be positive.")
        return result

    distribution = NormalTanhDistribution(event_size=action_size)
    policy = TrustMLP(layers("policy"), distribution.param_size)
    q_count = config.get("value", {}).get("num_networks", 1)
    if q_count < 1:
        raise ValueError("num_networks must be positive.")
    critic = TrustCritic(layers("value"), q_count)
    value = TrustMLP(layers("state_value"))
    obs, action = (jnp.zeros((1, observation_size)), jnp.zeros((1, action_size)))
    return TrustNetworks(
        Network(lambda k: policy.init(k, obs), policy.apply),
        Network(lambda k: critic.init(k, obs, action), critic.apply),
        Network(lambda k: value.init(k, obs), value.apply),
        distribution,
        optax.adam(learning_rate),
        optax.adam(learning_rate),
        optax.adam(learning_rate),
    )


def make_inference_fn(network: TrustNetworks):

    def make_policy(params, deterministic: bool = False):

        def policy(observations, key_sample=None):
            logits = network.policy_network.apply(params, observations)
            distribution = network.parametric_action_distribution
            action = (
                distribution.mode(logits)
                if deterministic
                else distribution.sample(logits, key_sample)
            )
            return (action, {})

        return policy

    return make_policy


def initialize(
    action_size: int,
    observation_size: int,
    learning_rate: float,
    network_config: dict,
    key: jax.Array,
):
    network = make_networks(
        observation_size, action_size, learning_rate=learning_rate, network_config=network_config
    )
    kp, kq, kv = jax.random.split(key, 3)
    policy = network.policy_network.init(kp)
    critic = network.value_network.init(kq)
    value = network.state_value_network.init(kv)
    state = TrustTrainingState(
        params=TrustNetworkParams(policy, critic, critic, value),
        policy_optimizer_state=network.policy_optimizer.init(policy),
        value_optimizer_state=network.value_optimizer.init(critic),
        state_value_optimizer_state=network.state_value_optimizer.init(value),
        rl_gradient_steps=jnp.array(0, dtype=jnp.int32),
    )
    return (network, state, make_inference_fn(network))


def make_sgd_step(
    network: TrustNetworks,
    discount: float,
    tau: float,
    beta: float,
    expectile: float,
    epsilon: float,
    iqr_alpha: float = 3.0,
    numerical_epsilon: float = 1e-06,
    axis_name: str | None = None,
):
    if (
        not 0 <= discount <= 1
        or not 0 < tau <= 1
        or beta <= 0
        or (not 0 < expectile < 1)
        or (epsilon < 0)
    ):
        raise ValueError("Invalid TRUST discount, Polyak coefficient, beta, expectile or epsilon.")
    if iqr_alpha < 0 or numerical_epsilon <= 0:
        raise ValueError("Invalid IQR coefficient or numerical stability constant.")

    def average_gradient(grads):
        return grads if axis_name is None else jax.lax.pmean(grads, axis_name)

    def step(carry, batch: datatypes.RLTransition):
        state, key = carry
        params = state.params
        value_fn = lambda obs: network.state_value_network.apply(params.state_value, obs)
        standard, robust = critic_targets(
            value_fn,
            batch.next_observation,
            batch.reward,
            batch.flag,
            batch.extras["trajectory_mask"],
            discount,
            epsilon,
        )

        def q_loss(q_params):
            q = network.value_network.apply(q_params, batch.observation, batch.action)
            per_sample = per_transition_critic_loss(q, standard, robust)
            weight = iqr_weights(per_sample, iqr_alpha, numerical_epsilon, axis_name)
            return (
                jnp.mean(weight * per_sample),
                {"iqr_weight_mean": jnp.mean(weight), "iqr_outlier_fraction": jnp.mean(weight > 1)},
            )

        (q_loss_value, auxiliary), q_grad = jax.value_and_grad(q_loss, has_aux=True)(params.value)
        q_values = network.value_network.apply(params.value, batch.observation, batch.action)
        q_values = jax.lax.stop_gradient(jnp.min(q_values, axis=-1))

        def v_loss(v_params):
            values = network.state_value_network.apply(v_params, batch.observation).reshape(-1)
            return jnp.mean(expectile_loss(q_values - values, expectile))

        v_loss_value, v_grad = jax.value_and_grad(v_loss)(params.state_value)
        values = value_fn(batch.observation).reshape(-1)
        advantages = jax.lax.stop_gradient(q_values - values)

        def policy_loss(policy_params):
            logits = network.policy_network.apply(policy_params, batch.observation)
            log_prob = squashed_gaussian_log_prob(
                network.parametric_action_distribution, logits, batch.action
            )
            return awr_loss(advantages, log_prob, beta)

        p_loss_value, p_grad = jax.value_and_grad(policy_loss)(params.policy)

        def apply(optimizer, gradient, optimizer_state, old_params):
            updates, new_state = optimizer.update(
                average_gradient(gradient), optimizer_state, old_params
            )
            return (optax.apply_updates(old_params, updates), new_state)

        q, qo = apply(network.value_optimizer, q_grad, state.value_optimizer_state, params.value)
        v, vo = apply(
            network.state_value_optimizer,
            v_grad,
            state.state_value_optimizer_state,
            params.state_value,
        )
        p, po = apply(network.policy_optimizer, p_grad, state.policy_optimizer_state, params.policy)
        target = optax.incremental_update(q, params.target_value, tau)
        updated = state.replace(
            params=TrustNetworkParams(p, q, target, v),
            policy_optimizer_state=po,
            value_optimizer_state=qo,
            state_value_optimizer_state=vo,
            rl_gradient_steps=state.rl_gradient_steps + 1,
        )
        metrics = {
            "value_loss": q_loss_value,
            "state_value_loss": v_loss_value,
            "policy_loss": p_loss_value,
            "robust_target_mean": jnp.mean(robust),
            **auxiliary,
        }
        if axis_name is not None:
            metrics = jax.lax.pmean(metrics, axis_name)
        return ((updated, jax.random.split(key)[0]), metrics)

    return step
