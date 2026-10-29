import numpy as np
from vmax.agents import datatypes
from vmax.agents.learning.offline_dataset import OfflineDataset


class OfflineReplayBuffer:

    def __init__(self, dataset: OfflineDataset, seed: int = 0):
        self.dataset = dataset
        self.size = dataset.size
        self._rng = np.random.default_rng(seed)
        self._flags = dataset.flags
        self._dones = dataset.dones

    def sample(self, batch_size: int) -> datatypes.RLTransition:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        indices = self._rng.integers(0, self.size, size=batch_size)
        ds = self.dataset
        mask = ds.trajectory_mask
        mask = (
            np.broadcast_to(mask, (batch_size, mask.shape[-1])) if mask.ndim == 1 else mask[indices]
        )
        return datatypes.RLTransition(
            observation=ds.observations[indices],
            action=ds.actions[indices],
            reward=ds.rewards[indices],
            flag=self._flags[indices],
            next_observation=ds.next_observations[indices],
            done=self._dones[indices],
            extras={"trajectory_mask": mask},
        )
