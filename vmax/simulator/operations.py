import jax
import jax.numpy as jnp
from waymax.datatypes import route


def get_index(x: jnp.ndarray, k: int = 1, squeeze: bool = True) -> jnp.ndarray:
    if k == 1:
        idx = jnp.argmax(x, keepdims=not squeeze)
    else:
        idx = jax.lax.top_k(x, k)[1]
        if squeeze:
            return idx.squeeze()
    return idx


def select_longest_sdc_path_id(sdc_paths: route.Paths) -> int:
    on_route = sdc_paths.on_route
    on_route = jnp.repeat(on_route, sdc_paths.num_points_per_path, axis=-1)
    mask = jnp.logical_and(on_route, sdc_paths.valid)
    longest_path_idx = jnp.argmax(jnp.sum(mask, axis=-1))
    return longest_path_idx
