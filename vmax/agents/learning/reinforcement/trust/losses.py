from collections.abc import Callable
import jax
import jax.numpy as jnp


def expectile_loss(residual: jax.Array, expectile: float = 0.8) -> jax.Array:
    return jnp.where(residual >= 0, expectile, 1 - expectile) * jnp.square(residual)


def awr_loss(
    advantage: jax.Array, log_prob: jax.Array, beta: float = 2.0, max_weight: float = 100.0
) -> jax.Array:
    advantage, log_prob = (advantage.reshape(-1), log_prob.reshape(-1))
    if advantage.shape != log_prob.shape:
        raise ValueError("AWR requires one advantage and one log probability per transition.")
    weight = jnp.exp(jnp.minimum(jax.lax.stop_gradient(advantage) / beta, jnp.log(max_weight)))
    return -jnp.mean(weight * log_prob)


def iqr_weights(
    losses: jax.Array,
    alpha: float = 3.0,
    numerical_epsilon: float = 1e-06,
    axis_name: str | None = None,
) -> jax.Array:
    losses = jax.lax.stop_gradient(losses.reshape(-1))
    batch = losses if axis_name is None else jax.lax.all_gather(losses, axis_name).reshape(-1)
    mean, std = (jnp.mean(batch), jnp.std(batch))
    scale = jnp.where(std > 0, std, 1.0)
    z_batch, z_local = ((batch - mean) / scale, (losses - mean) / scale)
    q25, q75 = jnp.percentile(z_batch, jnp.array([25.0, 75.0]))
    fence = q75 + alpha * (q75 - q25)
    denominator = jnp.abs(fence) + numerical_epsilon
    denominator = jnp.where(denominator > 0, denominator, 1.0)
    return jax.lax.stop_gradient(
        jnp.where(z_local > fence, 1 + (z_local - fence) / denominator, 1.0)
    )


def critic_targets(
    value_fn: Callable,
    next_observation: jax.Array,
    reward: jax.Array,
    flag: jax.Array,
    trajectory_mask: jax.Array,
    discount: float,
    epsilon: float,
):

    def single_value(obs):
        return value_fn(obs[None, :]).reshape(())

    values = value_fn(next_observation).reshape(-1)
    gradients = jax.vmap(jax.grad(single_value))(next_observation)
    norm = jnp.sum(jnp.abs(gradients) * trajectory_mask, axis=-1)
    standard = reward.reshape(-1) + discount * flag.reshape(-1) * values
    robust = reward.reshape(-1) + discount * flag.reshape(-1) * (values - epsilon * norm)
    return (jax.lax.stop_gradient(standard), jax.lax.stop_gradient(robust))


def per_transition_critic_loss(q: jax.Array, standard: jax.Array, robust: jax.Array) -> jax.Array:
    q = q.reshape((len(standard), -1))
    return jnp.mean(0.5 * ((q - standard[:, None]) ** 2 + (q - robust[:, None]) ** 2), axis=-1)


def squashed_gaussian_log_prob(distribution, parameters: jax.Array, action: jax.Array) -> jax.Array:
    action = jnp.clip(action, -1 + 1e-06, 1 - 1e-06)
    return distribution.log_prob(parameters, distribution.inverse_postprocess(action))
