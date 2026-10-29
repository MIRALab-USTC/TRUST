from abc import ABC, abstractmethod
import jax
from waymax import datatypes


class AbstractFeaturesExtractor(ABC):

    @abstractmethod
    def extract_features(self, state: datatypes.SimulatorState) -> tuple[jax.Array, ...]:
        pass

    @abstractmethod
    def unflatten_features(
        self, vectorized_obs: jax.Array
    ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        pass
