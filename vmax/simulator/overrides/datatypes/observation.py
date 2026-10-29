__copyright__ = "Copyright 2023 The Waymax Authors."
__license__ = "Waymax License Agreement for Non-commercial Use"
__modifications__ = "Adapted for vector observations, roadgraph filtering, and SDC path transforms."

import chex
import jax
import jax.numpy as jnp
from waymax import config
from waymax.datatypes import route, simulator_state
from waymax.datatypes.observation import (
    ObjectPose2D,
    Observation,
    _tree_expand_and_repeat,
    combine_two_object_pose_2d,
    global_observation_from_state,
    transform_roadgraph_points,
    transform_traffic_lights,
    transform_trajectory,
)
from waymax.utils import geometry
from .roadgraph import filter_box_roadgraph_points, filter_topk_roadgraph_points
from .route import filter_sdc_paths


def transform_sdc_paths(paths: route.Paths, pose2d: ObjectPose2D) -> route.Paths:
    chex.assert_equal(paths.shape, pose2d.shape + paths.shape[-2:])
    dst_xy = geometry.transform_points(pts=paths.xy, pose_matrix=pose2d.matrix)
    valid = jnp.logical_and(paths.valid, pose2d.valid[..., None, None])
    return paths.replace(x=dst_xy[..., 0], y=dst_xy[..., 1], valid=valid)


def transform_observation(observation: Observation, pose2d: ObjectPose2D) -> Observation:
    chex.assert_equal_shape([observation, pose2d])
    pose = combine_two_object_pose_2d(src_pose=observation.pose2d, dst_pose=pose2d)
    transf_traj = transform_trajectory(observation.trajectory, pose)
    transf_rg = transform_roadgraph_points(observation.roadgraph_static_points, pose)
    transf_traj = transform_trajectory(observation.trajectory, pose)
    transf_tls = transform_traffic_lights(observation.traffic_lights, pose)
    if observation.sdc_paths is not None:
        transf_sdc_paths = transform_sdc_paths(observation.sdc_paths, pose)
    obs = observation.replace(
        trajectory=transf_traj,
        roadgraph_static_points=transf_rg,
        traffic_lights=transf_tls,
        sdc_paths=transf_sdc_paths,
        pose2d=pose2d,
    )
    obs.validate()
    return obs


@jax.named_scope("sdc_observation_from_state")
def sdc_observation_from_state(
    state: simulator_state.SimulatorState,
    obs_num_steps: int = 1,
    roadgraph_top_k: int | None = 1000,
    meters_box: dict[int] | None = None,
    coordinate_frame: config.CoordinateFrame = config.CoordinateFrame.SDC,
) -> Observation:
    obj_xy = state.current_sim_trajectory.xy[..., 0, :]
    obj_yaw = state.current_sim_trajectory.yaw[..., 0]
    obj_valid = state.current_sim_trajectory.valid[..., 0]
    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
    sdc_xy = jnp.take_along_axis(obj_xy, sdc_idx[..., jnp.newaxis], axis=-2)
    sdc_yaw = jnp.take_along_axis(obj_yaw, sdc_idx, axis=-1)
    sdc_valid = jnp.take_along_axis(obj_valid, sdc_idx, axis=-1)
    num_obj = 1
    global_obs = global_observation_from_state(state, obs_num_steps, num_obj=num_obj)
    is_ego = state.object_metadata.is_sdc[..., jnp.newaxis, :]
    if meters_box is None:
        global_obs_filter = global_obs.replace(
            is_ego=is_ego,
            roadgraph_static_points=filter_topk_roadgraph_points(
                global_obs.roadgraph_static_points, sdc_xy, roadgraph_top_k
            ),
        )
    else:
        global_obs_filter = global_obs.replace(
            is_ego=is_ego,
            roadgraph_static_points=filter_box_roadgraph_points(
                global_obs.roadgraph_static_points, sdc_xy, sdc_yaw, meters_box, roadgraph_top_k
            ),
        )
    if state.sdc_paths is not None:
        sdc_paths = state.sdc_paths
        num_points_per_path_filter = min(299, sdc_paths.num_points_per_path)
        sdc_paths_expanded = _tree_expand_and_repeat(sdc_paths, num_obj, len(state.shape))
        sdc_paths_expanded = jax.tree_map(lambda x: x.swapaxes(0, 1), sdc_paths_expanded)
        sdc_paths_filtered = jax.vmap(filter_sdc_paths, in_axes=(0, None, None))(
            sdc_paths_expanded, sdc_xy, num_points_per_path_filter
        )
        sdc_paths_filtered = jax.tree_map(lambda x: x.swapaxes(0, 1), sdc_paths_filtered)
        global_obs_filter = global_obs_filter.replace(sdc_paths=sdc_paths_filtered)
        global_obs_filter.validate()
    if coordinate_frame in (config.CoordinateFrame.OBJECT, config.CoordinateFrame.SDC):
        pose2d = ObjectPose2D.from_center_and_yaw(xy=sdc_xy, yaw=sdc_yaw, valid=sdc_valid)
        chex.assert_equal(pose2d.shape, state.shape + (1,))
        return transform_observation(global_obs_filter, pose2d)
    elif coordinate_frame == config.CoordinateFrame.GLOBAL:
        return global_obs_filter
    else:
        raise NotImplementedError(f"Coordinate frame {coordinate_frame} not supported.")
