from typing import Any
import chex
import jax
import jax.numpy as jnp
from flax import struct
from waymax import datatypes
from waymax import env as waymax_env
from waymax.datatypes import operations
from waymax.env import typedefs as types
from vmax.simulator.wrappers.base import Wrapper


@chex.dataclass(frozen=True)
class EnvTransition:
    state: datatypes.SimulatorState
    observation: types.Observation
    reward: jax.Array
    done: jax.Array
    flag: jax.Array
    metrics: dict[str, Any] = struct.field(default_factory=dict)
    info: dict[str, Any] = struct.field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.state.shape

    def __eq__(self, other: object) -> bool:
        return datatypes.compare_all_leaf_nodes(self, other)


class BraxWrapper(Wrapper):

    def __init__(
        self,
        env: waymax_env.PlanningAgentEnvironment,
        termination_keys: list[str] = ("offroad", "overlap"),
    ) -> None:
        super().__init__(env)
        self._termination_keys = termination_keys

    def termination(self, state: datatypes.SimulatorState) -> jax.Array:
        metrics = self.env.metrics(state)
        termination = 0
        for key in self._termination_keys:
            termination += metrics[key].value
        return jnp.array(termination > 0, dtype=jnp.int32)

    def metrics(self, state: datatypes.SimulatorState) -> types.Metrics:
        metrics = self.env.metrics(state)
        metrics_dict = {key: metric.value for key, metric in metrics.items()}
        return metrics_dict

    def reset(self, state: datatypes.SimulatorState, rng: jax.Array | None = None) -> EnvTransition:
        chex.assert_equal(state.shape, ())
        state = self.env.reset(state, rng)
        obs = self.observe(state)
        reward = jnp.zeros(state.shape + self.discount_spec().shape, dtype=jnp.float32)
        flag = jnp.ones(state.shape + self.discount_spec().shape, dtype=jnp.bool_)
        done = jnp.zeros(state.shape + self.discount_spec().shape, dtype=jnp.bool_)
        metric_dict = self.metrics(state)
        info = {
            "steps": jnp.zeros(state.shape, dtype=jnp.int32),
            "rewards": jnp.zeros(state.shape, dtype=jnp.float32),
            "truncation": jnp.zeros(state.shape, dtype=jnp.bool_),
            "scenario_id": jnp.zeros(state.shape, dtype=jnp.int32),
        }
        return EnvTransition(
            state=state,
            observation=obs,
            reward=reward,
            done=done,
            flag=flag,
            metrics=metric_dict,
            info=info,
        )

    def step(self, env_transition: EnvTransition, action: datatypes.Action) -> EnvTransition:
        chex.assert_equal(env_transition.state.shape, ())
        next_state = self.env.step(env_transition.state, action)
        obs = self.observe(next_state)
        reward = self.reward(next_state, action)
        termination = self.termination(next_state)
        truncation = self.truncation(next_state)
        done = jnp.logical_or(termination, truncation)
        flag = jnp.logical_not(termination)
        metric_dict = self.metrics(next_state)
        info = {
            "steps": env_transition.info["steps"] + 1,
            "rewards": env_transition.info["rewards"] + reward,
            "truncation": truncation,
            "scenario_id": env_transition.info["scenario_id"],
        }
        return EnvTransition(
            state=next_state,
            reward=reward,
            observation=obs,
            done=done,
            flag=flag,
            metrics=metric_dict,
            info=info,
        )


class VmapWrapper(Wrapper):

    def reset(self, state: datatypes.SimulatorState, rng: jax.Array | None = None) -> EnvTransition:
        return jax.vmap(self.env.reset)(state, rng)

    def step(self, env_transition: EnvTransition, action: jax.Array) -> EnvTransition:
        return jax.vmap(self.env.step)(env_transition, action)
