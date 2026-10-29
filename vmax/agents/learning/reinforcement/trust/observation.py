import numpy as np

_WIDTHS = {
    "waypoints": 2,
    "velocity": 2,
    "speed": 1,
    "yaw": 1,
    "size": 2,
    "valid": 1,
    "direction": 2,
    "types": 8,
    "state": 8,
    "object_types": 4,
}
_TRAJECTORY = {"waypoints", "velocity", "speed", "yaw"}


def _width(features):
    try:
        return sum((_WIDTHS[f] for f in features))
    except KeyError as error:
        raise ValueError(f"Unsupported observation feature: {error.args[0]}") from error


def observation_layout(config: dict):
    history = config.get("obs_past_num_steps", 1)
    objects = config.get("objects", {})
    fields = objects.get("features", [])
    count = 1 + objects.get("num_closest_objects", 8)
    width = _width(fields)
    object_dim = count * history * width
    road = config.get("roadgraphs", {})
    lights = config.get("traffic_lights", {})
    path = config.get("path_target", {})
    total = object_dim + road.get("roadgraph_top_k", 200) * _width(road.get("features", []))
    total += (
        lights.get("num_closest_traffic_lights", 5) * history * _width(lights.get("features", []))
    )
    total += path.get("num_points", 3) * _width(path.get("features", []))
    coordinate_mask = np.zeros(width, dtype=bool)
    offset, valid_index = (0, None)
    for field in fields:
        size = _WIDTHS[field]
        if field in _TRAJECTORY:
            coordinate_mask[offset : offset + size] = True
        if field == "valid":
            valid_index = offset
        offset += size
    return (total, count, history, width, coordinate_mask, valid_index)


def build_trajectory_mask(
    observations: np.ndarray, config: dict, vehicle_mask: np.ndarray
) -> np.ndarray:
    observations = np.asarray(observations)
    total, count, history, width, coordinates, valid_index = observation_layout(config)
    if observations.ndim != 2 or observations.shape[-1] != total:
        raise ValueError(
            f"Observation dimension does not match observation_config (expected {total})."
        )
    vehicle_mask = np.asarray(vehicle_mask)
    if not np.isin(vehicle_mask, [0, 1]).all():
        raise ValueError("vehicle_mask must be binary.")
    if vehicle_mask.shape == (len(observations), count):
        vehicle_mask = vehicle_mask[..., None]
    if vehicle_mask.shape not in (
        (len(observations), count, 1),
        (len(observations), count, history),
    ):
        raise ValueError("vehicle_mask must have shape (N,K) or (N,K,H).")
    dim = count * history * width
    blocks = observations[:, :dim].reshape(-1, count, history, width)
    valid = (
        np.ones(blocks.shape[:-1], dtype=bool)
        if valid_index is None
        else blocks[..., valid_index] > 0.5
    )
    selected = valid & vehicle_mask.astype(bool)
    fields = config.get("objects", {}).get("features", [])
    if "object_types" in fields:
        type_offset = _width(fields[: fields.index("object_types")])
        selected &= blocks[..., type_offset] > 0.5
    mask = np.zeros(observations.shape, dtype=bool)
    mask[:, :dim] = (selected[..., None] & coordinates).reshape(len(observations), -1)
    return mask


def validate_trajectory_mask(observations: np.ndarray, mask: np.ndarray, config: dict) -> None:
    _, count, _, _, _, _ = observation_layout(config)
    for start in range(0, len(observations), 8192):
        obs = observations[start : start + 8192]
        allowed = build_trajectory_mask(obs, config, np.ones((len(obs), count), dtype=bool))
        actual = mask if mask.ndim == 1 else mask[start : start + len(obs)]
        if np.any(actual & ~allowed):
            raise ValueError(
                "trajectory_mask selects nontrajectory or invalid coordinates; regenerate the saved mask."
            )
