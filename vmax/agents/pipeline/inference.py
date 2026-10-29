import jax.numpy as jnp
from waymax import datatypes


def policy_step(env_transition, env, policy_fn, key=None):
    actions, _ = policy_fn(env_transition.observation, key)
    action = datatypes.Action(data=actions, valid=jnp.ones_like(actions[..., :1], dtype=bool))
    action.validate()
    return env.step(env_transition, action), None
