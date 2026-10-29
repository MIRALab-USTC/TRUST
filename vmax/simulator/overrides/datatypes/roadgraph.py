__copyright__ = "Copyright 2023 The Waymax Authors."
__license__ = "Waymax License Agreement for Non-commercial Use"
__modifications__ = "Adapted for rectangular roadgraph filtering and lane point ordering."

import chex
import jax
from jax import numpy as jnp
from waymax.datatypes.roadgraph import RoadgraphPoints


def rotate_rectangle(rectangle: jnp.ndarray, yaw: jnp.ndarray) -> jnp.ndarray:
    rot_matrix = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw)], [jnp.sin(yaw), jnp.cos(yaw)]])
    return jnp.dot(rectangle, rot_matrix)


def points_in_rectangle(points: jnp.ndarray, rectangle: jnp.ndarray) -> jnp.ndarray:
    edge1 = rectangle[1] - rectangle[0]
    edge2 = rectangle[3] - rectangle[0]
    edge1_normalized = edge1 / jnp.linalg.norm(edge1)
    edge2_normalized = edge2 / jnp.linalg.norm(edge2)
    points_local = points - rectangle[0]
    proj1 = jnp.dot(points_local, edge1_normalized)
    proj2 = jnp.dot(points_local, edge2_normalized)
    in_bounds = jnp.logical_and(
        jnp.logical_and(proj1 >= 0, proj1 <= jnp.linalg.norm(edge1)),
        jnp.logical_and(proj2 >= 0, proj2 <= jnp.linalg.norm(edge2)),
    )
    return in_bounds


def filter_box_roadgraph_points(
    roadgraph: RoadgraphPoints,
    reference_points: jax.Array,
    reference_yaw: jax.Array,
    meters_box: list[int],
    topk: int,
) -> RoadgraphPoints:
    chex.assert_equal_shape_prefix([roadgraph, reference_points], reference_points.ndim - 1)
    chex.assert_equal(len(roadgraph.shape), reference_points.ndim)
    chex.assert_equal(reference_points.shape[-1], 2)
    reference_box = jnp.array(
        [
            [-meters_box["back"], -meters_box["right"]],
            [meters_box["front"], -meters_box["right"]],
            [meters_box["front"], meters_box["left"]],
            [-meters_box["back"], meters_box["left"]],
        ]
    )
    rotated_box = rotate_rectangle(reference_box, reference_yaw).squeeze()
    translated_box = rotated_box + reference_points
    distances = jnp.linalg.norm(reference_points[..., jnp.newaxis, :] - roadgraph.xy, axis=-1)
    roadgraph.valid = points_in_rectangle(roadgraph.xy, translated_box)
    valid_distances = jnp.where(roadgraph.valid, distances, float("inf"))
    _, top_idx = jax.lax.top_k(-valid_distances, topk)
    _idx = jnp.argsort(top_idx, axis=-1)
    top_idx = jnp.take_along_axis(top_idx, _idx, axis=-1)
    stacked = jnp.stack(
        [
            roadgraph.x,
            roadgraph.y,
            roadgraph.z,
            roadgraph.dir_x,
            roadgraph.dir_y,
            roadgraph.dir_z,
            roadgraph.types,
            roadgraph.ids,
            roadgraph.valid,
        ],
        axis=-1,
        dtype=jnp.float32,
    )
    filtered = jnp.take_along_axis(stacked, top_idx[..., None], axis=-2)
    return RoadgraphPoints(
        x=filtered[..., 0],
        y=filtered[..., 1],
        z=filtered[..., 2],
        dir_x=filtered[..., 3],
        dir_y=filtered[..., 4],
        dir_z=filtered[..., 5],
        types=filtered[..., 6].astype(jnp.int32),
        ids=filtered[..., 7].astype(jnp.int32),
        valid=filtered[..., 8].astype(jnp.bool_),
    )


def filter_topk_roadgraph_points(
    roadgraph: RoadgraphPoints, reference_points: jax.Array, topk: int
) -> RoadgraphPoints:
    chex.assert_equal_shape_prefix([roadgraph, reference_points], reference_points.ndim - 1)
    chex.assert_equal(len(roadgraph.shape), reference_points.ndim)
    chex.assert_equal(reference_points.shape[-1], 2)
    if topk > roadgraph.num_points:
        raise NotImplementedError("Not enough points in roadgraph.")
    elif topk < roadgraph.num_points:
        distances = jnp.linalg.norm(reference_points[..., jnp.newaxis, :] - roadgraph.xy, axis=-1)
        valid_distances = jnp.where(roadgraph.valid, distances, float("inf"))
        _, top_idx = jax.lax.top_k(-valid_distances, topk)
        _idx = jnp.argsort(top_idx, axis=-1)
        top_idx = jnp.take_along_axis(top_idx, _idx, axis=-1)
        stacked = jnp.stack(
            [
                roadgraph.x,
                roadgraph.y,
                roadgraph.z,
                roadgraph.dir_x,
                roadgraph.dir_y,
                roadgraph.dir_z,
                roadgraph.types,
                roadgraph.ids,
                roadgraph.valid,
            ],
            axis=-1,
            dtype=jnp.float32,
        )
        filtered = jnp.take_along_axis(stacked, top_idx[..., None], axis=-2)
        return RoadgraphPoints(
            x=filtered[..., 0],
            y=filtered[..., 1],
            z=filtered[..., 2],
            dir_x=filtered[..., 3],
            dir_y=filtered[..., 4],
            dir_z=filtered[..., 5],
            types=filtered[..., 6].astype(jnp.int32),
            ids=filtered[..., 7].astype(jnp.int32),
            valid=filtered[..., 8].astype(jnp.bool_),
        )
    else:
        return roadgraph
