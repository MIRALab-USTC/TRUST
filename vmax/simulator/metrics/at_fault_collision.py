import jax.numpy as jnp
from waymax import datatypes
from waymax.metrics import abstract_metric
from waymax.utils import geometry
from vmax.simulator import operations
from vmax.simulator.metrics import on_multiple_lanes, utils

LANE_WIDTH = 3.0


class AtFaultCollisionMetric(abstract_metric.AbstractMetric):

    def compute(self, simulator_state: datatypes.SimulatorState) -> abstract_metric.MetricResult:
        current_object_state = datatypes.dynamic_slice(
            simulator_state.sim_trajectory, simulator_state.timestep, 1, -1
        )
        sdc_index = operations.get_index(simulator_state.object_metadata.is_sdc)
        collisions, valid = self._collides_with_ego(current_object_state, sdc_index)
        rear_collisions, front_collisions, lateral_collisions = self._classify_collision(
            current_object_state, sdc_index
        )
        is_vru = jnp.isin(simulator_state.object_metadata.object_types, jnp.array([2, 3]))
        is_vru = jnp.logical_and(is_vru, valid)
        is_vru = jnp.logical_and(is_vru, collisions)
        rear_collisions = jnp.logical_and(rear_collisions, collisions)
        front_collisions = jnp.logical_and(front_collisions, collisions)
        lateral_collisions = jnp.logical_and(lateral_collisions, collisions)
        stopped = self._is_vehicle_stopped(current_object_state)
        ego_stopped = stopped[sdc_index]
        stopped = jnp.logical_and(stopped, collisions)
        ego_on_multiple_lanes = on_multiple_lanes.is_ego_on_multiple_lanes(simulator_state)
        at_fault = (
            is_vru
            + stopped
            + ~ego_stopped * (front_collisions + lateral_collisions * ego_on_multiple_lanes)
        )
        other_objects = jnp.arange(at_fault.shape[-1]) != sdc_index
        value = jnp.any(jnp.logical_and(other_objects, at_fault)).astype(jnp.float32)
        valid = jnp.ones_like(value, dtype=bool)
        result = abstract_metric.MetricResult.create_and_validate(value, valid)
        return result

    def _collides_with_ego(self, current_traj: datatypes.Trajectory, sdc_index: int):
        traj_5dof = current_traj.stack_fields(["x", "y", "length", "width", "yaw"])
        pairwise_overlap = geometry.compute_pairwise_overlaps(traj_5dof[..., 0, :])
        valid = current_traj.valid[..., 0:1].squeeze()
        collisions = pairwise_overlap[sdc_index]
        collisions = jnp.logical_and(collisions, valid)
        return (collisions, valid)

    def _classify_collision(self, current_traj: datatypes.Trajectory, sdc_index: int):
        ego_yaw = current_traj.yaw.squeeze()[sdc_index]
        ego_length = current_traj.length.squeeze()[sdc_index]
        positions = current_traj.stack_fields(["x", "y"]).squeeze()
        ego_direction = jnp.array([jnp.cos(ego_yaw), jnp.sin(ego_yaw)])
        ego_rear_axle = positions[sdc_index] - ego_length * 0.4 * ego_direction
        rear_collisions = utils.is_agent_behind(ego_rear_axle, ego_yaw, positions)
        traj_5dof = current_traj.stack_fields(["x", "y", "length", "width", "yaw"]).squeeze()
        ego_front_bumper = positions[sdc_index] + ego_length * 0.5 * ego_direction
        traj_5dof = traj_5dof.at[sdc_index, 0].set(ego_front_bumper[0])
        traj_5dof = traj_5dof.at[sdc_index, 1].set(ego_front_bumper[1])
        traj_5dof = traj_5dof.at[sdc_index, 2].set(0)
        bumper_collisions = geometry.compute_pairwise_overlaps(traj_5dof)
        front_collisions = bumper_collisions[sdc_index]
        lateral_collisions = ~(rear_collisions + front_collisions)
        return (rear_collisions, front_collisions, lateral_collisions)

    def _is_vehicle_stopped(self, current_traj):
        almost_zero = 0.05
        speeds = current_traj.speed.squeeze()
        stopped = speeds < almost_zero
        return stopped
