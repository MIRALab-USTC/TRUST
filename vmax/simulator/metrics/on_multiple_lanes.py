import jax.numpy as jnp
from waymax import datatypes
from waymax.metrics import abstract_metric
from waymax.utils import geometry
from vmax.simulator import constants, operations
from vmax.simulator.metrics import utils


class OnMultipleLanesMetric(abstract_metric.AbstractMetric):

    def compute(self, simulator_state: datatypes.SimulatorState) -> abstract_metric.MetricResult:
        sdc_index = operations.get_index(simulator_state.object_metadata.is_sdc)
        on_multiple_lanes = is_ego_on_multiple_lanes(simulator_state)
        ego_speed = simulator_state.current_sim_trajectory.speed[sdc_index].squeeze()
        value = ego_speed * on_multiple_lanes * constants.TIME_DELTA
        valid = jnp.ones_like(value, dtype=jnp.bool_)
        result = abstract_metric.MetricResult.create_and_validate(value.astype(jnp.float32), valid)
        return result


def is_ego_on_multiple_lanes(state: datatypes.SimulatorState):
    sdc_index = operations.get_index(state.object_metadata.is_sdc)
    traj_5dof = state.current_sim_trajectory.stack_fields(
        ["x", "y", "length", "width", "yaw"]
    ).squeeze()
    ego_xyz = state.current_sim_trajectory.xyz[sdc_index].squeeze()
    ego_corners = geometry.corners_from_bbox(traj_5dof[sdc_index])
    roadgraph_points = state.roadgraph_points
    corners_distance_to_lane_center = utils.get_corners_distance_to_closest_lane_center(
        ego_xyz, ego_corners, roadgraph_points
    )
    return jnp.max(corners_distance_to_lane_center) > utils.LANE_WIDTH / 2 + utils.MARGIN
