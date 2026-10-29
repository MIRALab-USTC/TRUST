from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class OfflineDataset:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    trajectory_mask: np.ndarray
    source_files: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.observations)

    @property
    def flags(self) -> np.ndarray:
        return (~self.terminals).astype(np.float32)

    @property
    def dones(self) -> np.ndarray:
        return np.logical_or(self.terminals, self.timeouts)


def _field(archive, name: str, *aliases: str, required: bool = True):
    present = [key for key in (name, *aliases) if key in archive]
    if not present:
        if required:
            raise ValueError(
                f"Missing '{name}' in offline data; complete saved transitions are required."
            )
        return None
    value = np.asarray(archive[present[0]])
    for key in present[1:]:
        if not np.array_equal(value, archive[key]):
            raise ValueError(f"Conflicting aliases for '{name}': {present}")
    return value


def _binary(value, name: str, size: int) -> np.ndarray:
    value = np.asarray(value)
    if value.shape == (size, 1):
        value = value[:, 0]
    if value.shape != (size,) or not np.isin(value, [0, 1]).all():
        raise ValueError(f"'{name}' must be a binary vector of length {size}.")
    return value.astype(bool)


def _read_shard(path: Path) -> OfflineDataset:
    with np.load(path, allow_pickle=False) as archive:
        obs = _field(archive, "observations", "observation", "obs")
        nxt = _field(archive, "next_observations", "next_observation", "next_obs")
        actions = _field(archive, "actions", "action")
        rewards = _field(archive, "rewards", "reward")
        mask = _field(archive, "trajectory_mask")
        if obs.ndim != 2 or min(obs.shape) == 0:
            raise ValueError("'observations' must have nonempty shape (N, observation_size).")
        size, dim = obs.shape
        if nxt.shape != obs.shape:
            raise ValueError("'next_observations' must have the same shape as observations.")
        if actions.ndim != 2 or actions.shape[0] != size or actions.shape[1] == 0:
            raise ValueError("'actions' must have shape (N, action_size).")
        if rewards.shape == (size, 1):
            rewards = rewards[:, 0]
        if rewards.shape != (size,):
            raise ValueError("'rewards' must have shape (N,) or (N, 1).")
        for name, value in (
            ("observations", obs),
            ("next_observations", nxt),
            ("actions", actions),
            ("rewards", rewards),
        ):
            if (
                not np.issubdtype(value.dtype, np.number)
                or np.iscomplexobj(value)
                or (not np.isfinite(value).all())
            ):
                raise ValueError(f"'{name}' must contain finite real numeric values.")
            limit = np.finfo(np.float32).max
            if np.min(value) < -limit or np.max(value) > limit:
                raise ValueError(f"'{name}' is not representable as finite float32 data.")
        if np.any(np.abs(actions) > 1.0 + 1e-06):
            raise ValueError("Actions must be normalized to [-1, 1] for the tanh policy.")
        if mask.shape not in ((dim,), (size, dim)) or not np.isin(mask, [0, 1]).all():
            raise ValueError("'trajectory_mask' must be binary, with shape (D,) or (N,D).")
        terminals_raw = _field(archive, "terminals", "terminated", required=False)
        timeouts_raw = _field(archive, "timeouts", "truncated", required=False)
        flag_raw = _field(archive, "flag", "flags", required=False)
        done_raw = _field(archive, "done", "dones", required=False)
        if terminals_raw is None:
            if flag_raw is None or done_raw is None:
                raise ValueError(
                    "Supply terminals/timeouts, or both legacy flag and done; done alone is ambiguous."
                )
            flags = _binary(flag_raw, "flag", size)
            dones = _binary(done_raw, "done", size)
            terminals = ~flags
            timeouts = dones & flags
        else:
            if timeouts_raw is None:
                raise ValueError("Explicit 'timeouts' is required alongside terminals.")
            terminals = _binary(terminals_raw, "terminals", size)
            timeouts = _binary(timeouts_raw, "timeouts", size)
        if flag_raw is not None and (
            not np.array_equal(_binary(flag_raw, "flag", size), ~terminals)
        ):
            raise ValueError("flag conflicts with terminal flags.")
        if done_raw is not None and (
            not np.array_equal(_binary(done_raw, "done", size), terminals | timeouts)
        ):
            raise ValueError("done conflicts with terminal/truncation flags.")
        if timeouts_raw is not None and (
            not np.array_equal(_binary(timeouts_raw, "timeouts", size), timeouts)
        ):
            raise ValueError("timeouts conflicts with legacy flag/done.")
        values = [np.asarray(x, dtype=np.float32) for x in (obs, actions, rewards, nxt)]
        if any((not np.isfinite(value).all() for value in values)):
            raise ValueError("Nonfinite data after float32 conversion.")
        values += [terminals, timeouts, mask.astype(bool)]
    for value in values:
        value.setflags(write=False)
    return OfflineDataset(*values, source_files=(str(path.resolve()),))


def load_offline_dataset(path: str | Path, expected_size: int | None = None) -> OfflineDataset:
    path = Path(path).expanduser()
    paths = sorted(path.glob("*.npz")) if path.is_dir() else [path]
    if not paths or any((not p.is_file() for p in paths)):
        raise FileNotFoundError(f"No saved offline dataset found at {path}.")
    if any((p.suffix.lower() != ".npz" for p in paths)):
        raise ValueError("Expected a saved .npz transition archive or directory of .npz shards.")
    shards = [_read_shard(p) for p in paths]
    reference = shards[0]
    for shard in shards[1:]:
        if (
            shard.observations.shape[1] != reference.observations.shape[1]
            or shard.actions.shape[1] != reference.actions.shape[1]
        ):
            raise ValueError("All offline shards must share observation and action dimensions.")
    if len(shards) == 1:
        result = reference
    else:
        names = ("observations", "actions", "rewards", "next_observations", "terminals", "timeouts")
        values = [np.concatenate([getattr(s, name) for s in shards]) for name in names]
        masks = [s.trajectory_mask for s in shards]
        if all((m.ndim == 1 and np.array_equal(m, masks[0]) for m in masks)):
            mask = masks[0]
        else:
            mask = np.concatenate(
                [np.broadcast_to(s.trajectory_mask, s.observations.shape) for s in shards]
            )
        values.append(mask)
        for value in values:
            value.setflags(write=False)
        result = OfflineDataset(*values, source_files=tuple((str(p.resolve()) for p in paths)))
    if expected_size is not None and result.size != expected_size:
        raise ValueError(
            f"Expected {expected_size:,} transitions, found {result.size:,}; no rows were discarded."
        )
    return result
