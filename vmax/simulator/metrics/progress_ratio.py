import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax.metrics import abstract_metric
from vmax.simulator import operations


class ProgressRatioMetric(abstract_metric.AbstractMetric):

    def compute(self, simulator_state: datatypes.SimulatorState) -> abstract_metric.MetricResult:
        sdc_index = operations.get_index(simulator_state.object_metadata.is_sdc)
        sdc_traj = jax.tree_map(lambda x: x[sdc_index], simulator_state.sim_trajectory)
        expert_traj = jax.tree_map(lambda x: x[sdc_index], simulator_state.log_trajectory)
        value = progress_ratio(sdc_traj, expert_traj)
        valid = jnp.ones_like(value, dtype=jnp.bool_)
        result = abstract_metric.MetricResult.create_and_validate(value, valid)
        return result


def progress_ratio(sdc_traj: datatypes.Trajectory, expert_traj: datatypes.Trajectory):
    sdc_traj = jax.tree_map(lambda x: x[10:], sdc_traj)
    expert_traj = jax.tree_map(lambda x: x[10:], expert_traj)
    expert_displacement = jnp.diff(expert_traj.stack_fields(["x", "y"]), axis=0)
    expert_dist = jnp.linalg.norm(expert_displacement, axis=-1)
    expert_cum_dist = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(expert_dist)])

    def closest_point_dist(sdc_xy):
        dists = jnp.linalg.norm(sdc_xy - expert_traj.xy, axis=-1)
        min_idx = jnp.argmin(dists)
        return expert_cum_dist[min_idx]

    sdc_progress = jax.vmap(closest_point_dist)(sdc_traj.xy)
    sdc_progress = jnp.where(sdc_traj.valid, sdc_progress, 0.0)
    progress_ratio = jnp.max(sdc_progress) / expert_cum_dist[-1]
    progress_ratio = jnp.where(expert_cum_dist[-1] < 0.5, 1.0, progress_ratio)
    return progress_ratio
