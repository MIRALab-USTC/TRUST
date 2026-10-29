__copyright__ = "Copyright 2023 The Waymax Authors."
__license__ = "Waymax License Agreement for Non-commercial Use"
__modifications__ = "Adapted for reference-point route filtering."

import chex
import jax
from jax import numpy as jnp
from waymax import datatypes


def filter_sdc_paths(
    sdc_paths: datatypes.Paths, reference_points: jax.Array, num_points: int
) -> datatypes.Paths:
    chex.assert_equal_shape_prefix([sdc_paths, reference_points], reference_points.ndim - 1)
    chex.assert_equal(len(sdc_paths.shape), reference_points.ndim)
    chex.assert_equal(reference_points.shape[-1], 2)
    if num_points > sdc_paths.num_points_per_path:
        raise NotImplementedError("Not enough points in roadgraph.")
    elif num_points < sdc_paths.num_points_per_path:
        distances = jnp.linalg.norm(reference_points[..., jnp.newaxis, :] - sdc_paths.xy, axis=-1)
        sdc_idx = jnp.argmin(distances, axis=-1)
        indices = jnp.arange(num_points) + sdc_idx + 1
        stacked = jnp.stack(
            [
                sdc_paths.x,
                sdc_paths.y,
                sdc_paths.z,
                sdc_paths.ids,
                sdc_paths.valid,
                sdc_paths.arc_length,
            ],
            axis=-1,
            dtype=jnp.float32,
        )
        filtered = jnp.take_along_axis(stacked, indices[None, ..., None], axis=-2, fill_value=-1.0)
        return datatypes.Paths(
            x=filtered[..., 0],
            y=filtered[..., 1],
            z=filtered[..., 2],
            ids=filtered[..., 3],
            valid=filtered[..., 4],
            arc_length=filtered[..., 5],
            on_route=sdc_paths.on_route,
        )
    else:
        return sdc_paths
