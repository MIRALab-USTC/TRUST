__copyright__ = "Copyright 2023 The Waymax Authors."
__license__ = "Waymax License Agreement for Non-commercial Use"
__modifications__ = "Adapted for TRUST's scenario route evaluation."

from jax import numpy as jnp
from waymax import datatypes
from waymax.metrics import abstract_metric
from vmax.simulator.metrics import utils


class OffRouteMetric(abstract_metric.AbstractMetric):
    MAX_DISTANCE_TO_ROUTE_PATH = utils.LANE_WIDTH / 2 + utils.MARGIN
    MAX_RELATIVE_DISTANCE_TO_OFF_ROUTE_PATH = MAX_DISTANCE_TO_ROUTE_PATH / 2 + utils.MARGIN

    def compute(self, simulator_state: datatypes.SimulatorState) -> abstract_metric.MetricResult:
        sdc_paths = simulator_state.sdc_paths
        if sdc_paths is None:
            raise ValueError("SimulatorState.sdc_paths required to compute the off-route metric.")
        obj_xy = datatypes.dynamic_slice(
            simulator_state.sim_trajectory.xy, simulator_state.timestep, 1, axis=-2
        )
        sdc_xy = datatypes.select_by_onehot(
            obj_xy[..., 0, :], simulator_state.object_metadata.is_sdc, keepdims=False
        )
        obj_valid = datatypes.dynamic_slice(
            simulator_state.sim_trajectory.valid, simulator_state.timestep, 1, axis=-1
        )
        sdc_valid = datatypes.select_by_onehot(
            obj_valid[..., 0], simulator_state.object_metadata.is_sdc, keepdims=False
        )
        sdc_dist_to_paths = jnp.linalg.norm(
            sdc_xy[..., jnp.newaxis, jnp.newaxis, :] - sdc_paths.xy, axis=-1, keepdims=False
        )
        sdc_dist_to_valid_paths = jnp.where(sdc_paths.valid, sdc_dist_to_paths, jnp.inf)
        sdc_dist_to_valid_paths = jnp.where(
            jnp.expand_dims(sdc_valid, (-1, -2)), sdc_dist_to_valid_paths, jnp.inf
        )
        sdc_dist_to_valid_on_route_paths = jnp.where(
            sdc_paths.on_route, sdc_dist_to_valid_paths, jnp.inf
        )
        sdc_dist_to_valid_off_route_paths = jnp.where(
            ~sdc_paths.on_route, sdc_dist_to_valid_paths, jnp.inf
        )
        min_sdc_dist_to_valid_on_route_paths = jnp.min(
            sdc_dist_to_valid_on_route_paths, axis=(-1, -2)
        )
        min_sdc_dist_to_valid_off_route_paths = jnp.min(
            sdc_dist_to_valid_off_route_paths, axis=(-1, -2)
        )
        sdc_off_route = (min_sdc_dist_to_valid_on_route_paths > self.MAX_DISTANCE_TO_ROUTE_PATH) | (
            min_sdc_dist_to_valid_on_route_paths - min_sdc_dist_to_valid_off_route_paths
            > self.MAX_RELATIVE_DISTANCE_TO_OFF_ROUTE_PATH
        )
        off_route = jnp.where(sdc_off_route, min_sdc_dist_to_valid_on_route_paths, 0.0)
        valid = jnp.isfinite(off_route)
        off_route = jnp.where(valid, off_route, 0.0)
        return abstract_metric.MetricResult.create_and_validate(off_route, valid)
