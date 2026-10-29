from typing import NamedTuple
import jax


class RLTransition(NamedTuple):
    observation: jax.Array
    action: jax.Array
    reward: float
    flag: int
    next_observation: jax.Array
    done: int
    extras: tuple = ()
