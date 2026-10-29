from collections.abc import Sequence
from dataclasses import field
import chex
import jax
import jax.numpy as jnp


@chex.dataclass
class ObjectFeatures:
    field_names: Sequence[str]
    xy: jax.Array = field(default_factory=lambda: jnp.array(()))
    vel_xy: jax.Array = field(default_factory=lambda: jnp.array(()))
    yaw: jax.Array = field(default_factory=lambda: jnp.array(()))
    length: jax.Array = field(default_factory=lambda: jnp.array(()))
    width: jax.Array = field(default_factory=lambda: jnp.array(()))
    object_types: jax.Array = field(default_factory=lambda: jnp.array(()))
    valid: jax.Array = field(default_factory=lambda: jnp.array(()))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.stack_fields().shape

    @property
    def batch_dims(self) -> tuple[int, ...]:
        return self.shape[:-1]

    @property
    def num_objects(self) -> int:
        return self.shape[0]

    @property
    def num_past_observation(self) -> int:
        return self.shape[1]

    def stack_fields(self) -> jax.Array:
        if len(self.field_names) == 0:
            return jnp.array(())
        return jnp.concatenate(
            [getattr(self, field_name) for field_name in self.field_names], axis=-1
        )


@chex.dataclass
class RoadgraphFeatures:
    field_names: Sequence[str]
    xy: jax.Array = field(default_factory=lambda: jnp.array(()))
    dir_xy: jax.Array = field(default_factory=lambda: jnp.array(()))
    types: jax.Array = field(default_factory=lambda: jnp.array(()))
    ids: jax.Array = field(default_factory=lambda: jnp.array(()))
    valid: jax.Array = field(default_factory=lambda: jnp.array(()))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.stack_fields().shape

    def stack_fields(self) -> jax.Array:
        if len(self.field_names) == 0:
            return jnp.array(())
        return jnp.concatenate(
            [getattr(self, field_name) for field_name in self.field_names], axis=-1
        )


@chex.dataclass
class TrafficLightFeatures:
    field_names: Sequence[str]
    xy: jax.Array = field(default_factory=lambda: jnp.array(()))
    state: jax.Array = field(default_factory=lambda: jnp.array(()))
    ids: jax.Array = field(default_factory=lambda: jnp.array(()))
    valid: jax.Array = field(default_factory=lambda: jnp.array(()))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.stack_fields().shape

    @property
    def num_traffic_lights(self) -> int:
        return self.shape[0]

    def stack_fields(self) -> jax.Array:
        if len(self.field_names) == 0:
            return jnp.array(())
        return jnp.concatenate(
            [getattr(self, field_name) for field_name in self.field_names], axis=-1
        )


@chex.dataclass
class PathTargetFeatures:
    xy: jax.Array = field(default_factory=lambda: jnp.array(()))
    valid: jax.Array = field(default_factory=lambda: jnp.array(()))

    @property
    def data(self) -> jax.Array:
        return self.xy
