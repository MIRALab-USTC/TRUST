import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax import metrics as waymax_metrics
from waymax.env.planning_agent_environment import PlanningAgentEnvironment
from vmax.simulator import metrics, operations
from vmax.simulator.wrappers.base import Wrapper


class RewardLinearWrapper(Wrapper):

    def __init__(self, env: PlanningAgentEnvironment, reward_config: dict) -> None:
        super().__init__(env)
        self._reward_config = reward_config

    def reward(self, state: datatypes.SimulatorState, action: datatypes.Action) -> jax.Array:
        reward = 0.0
        for reward_name, reward_weigth in self._reward_config.items():
            reward_fn = _get_reward_fn(reward_name)
            reward += reward_fn(state) * reward_weigth
        return jnp.array(reward, dtype=jnp.float32)


def _get_reward_fn(reward_name: str) -> callable:
    reward_dict = {
        "overlap": _compute_overlap_reward,
        "offroad": _compute_offroad_reward,
        "red_light": _compute_red_light_reward,
        "off_route": _compute_off_route_reward,
        "progression": _compute_making_progress_reward,
    }
    if reward_name not in reward_dict:
        raise ValueError(f"Unsupported reward: {reward_name}")
    return reward_dict[reward_name]


def _compute_overlap_reward(state: datatypes.SimulatorState) -> bool:
    overlap = waymax_metrics.OverlapMetric().compute(state).value
    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    sdc_overlap = jax.tree_util.tree_map(lambda x: x[sdc_idx], overlap)
    return sdc_overlap == 1.0


def _compute_offroad_reward(state: datatypes.SimulatorState) -> bool:
    offroad = waymax_metrics.OffroadMetric().compute(state).value
    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    sdc_offroad = jax.tree_util.tree_map(lambda x: x[sdc_idx], offroad)
    return sdc_offroad == 1.0


def _compute_red_light_reward(state: datatypes.SimulatorState) -> bool:
    has_runned_red_light = metrics.RunRedLightMetric().compute(state).value
    return has_runned_red_light


def _compute_off_route_reward(state: datatypes.SimulatorState) -> bool:
    off_route_score = metrics.OffRouteMetric().compute(state).value
    return off_route_score > 0


def _compute_making_progress_reward(state: datatypes.SimulatorState) -> bool:
    current_progression = waymax_metrics.ProgressionMetric().compute(state).value
    n_state = state.replace(timestep=state.timestep - 1)
    previous_progression = waymax_metrics.ProgressionMetric().compute(n_state).value
    return current_progression > previous_progression
