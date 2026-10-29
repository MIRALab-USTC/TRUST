import jax
import jax.numpy as jnp

MAX_METERS = 50
MAX_SPEED = 30
RG_MAPPING = (0, 1, 2, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 5, 5, 6, 7, 8, 0)
TL_MAPPING = tuple(range(9))
OBJECT_MAPPING = tuple(range(5))


def get_feature_size(feature_key: str, dict_mapping: dict) -> int:
    if feature_key in ["xy", "vel_xy", "dir_xy"]:
        return 2
    elif feature_key in ["speed", "length", "width", "height", "valid", "yaw", "arc_length"]:
        return 1
    elif feature_key == "state":
        return max(dict_mapping["state"])
    elif feature_key == "types":
        return max(dict_mapping["types"])
    elif feature_key == "object_types":
        return max(dict_mapping["object_types"])
    else:
        raise ValueError(f"Feature {feature_key} not supported")


def normalize_path(x: jax.Array, meters: int) -> jax.Array:
    x = jnp.clip(x, min=-meters, max=meters)
    x = x / meters
    return x


def normalize_by_feature(
    data: jax.Array, feature_key: str, meters: int, dict_mapping: dict
) -> jax.Array:
    if feature_key == "xy":
        data = normalize_path(data, meters)
    elif feature_key == "state":
        data = onehot_encoder(data, dict_mapping["state"])
    elif feature_key == "types":
        data = onehot_encoder(data, dict_mapping["types"])
    elif feature_key == "object_types":
        data = onehot_encoder(data, dict_mapping["object_types"])
    elif feature_key == "vel_xy":
        data = jnp.clip(data, min=0, max=MAX_SPEED)
        data = data / MAX_SPEED
    elif feature_key in ["length", "width", "height"]:
        data = data / meters
    elif feature_key in ["valid", "yaw", "arc_length", "dir_xy", "ids"]:
        pass
    else:
        raise ValueError(f"Feature {feature_key} not supported")
    return data


def onehot_encoder(types: jax.Array, mapping: tuple[int]) -> jax.Array:
    mapped = jnp.take(jnp.array(mapping), types, axis=-1)
    onehot = jax.nn.one_hot(mapped, max(mapping) + 1, axis=-1)
    return onehot[..., 1:]
