import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax import env as waymax_env
from vmax.simulator.features import extractor
from vmax.simulator.wrappers.base import Wrapper


class ObservationWrapper(Wrapper):

    def __init__(
        self,
        env: waymax_env.PlanningAgentEnvironment,
        obs_type: str,
        obs_past_num_steps: int = 1,
        objects: dict | None = None,
        roadgraphs: dict | None = None,
        traffic_lights: dict | None = None,
        path_target: dict | None = None,
    ) -> None:
        super().__init__(env)
        self._features_extractor = extractor.get_extractor(obs_type)(
            obs_past_num_steps=obs_past_num_steps,
            objects_config=objects,
            roadgraphs_config=roadgraphs,
            traffic_lights_config=traffic_lights,
            path_target_config=path_target,
        )

    @property
    def features_extractor(self) -> extractor.AbstractFeaturesExtractor:
        return self._features_extractor

    def observation_spec(self) -> int:
        extractor = self.features_extractor
        object_features_size = extractor.get_features_size(extractor._object_features_key)
        roadgraph_features_size = extractor.get_features_size(extractor._roadgraph_features_key)
        traffic_lights_features_size = extractor.get_features_size(
            extractor._traffic_lights_features_key
        )
        path_target_feature_size = extractor.get_features_size(extractor._path_target_features_key)
        sdc_size = 1 * extractor._obs_past_num_steps * object_features_size
        other_objects_size = (
            extractor._num_closest_objects * extractor._obs_past_num_steps * object_features_size
        )
        roadgraph_size = extractor._roadgraph_top_k * roadgraph_features_size
        traffic_lights_size = (
            extractor._num_closest_traffic_lights
            * extractor._obs_past_num_steps
            * traffic_lights_features_size
        )
        path_target_size = extractor._num_target_path_points * path_target_feature_size
        return (
            sdc_size + other_objects_size + roadgraph_size + traffic_lights_size + path_target_size
        )

    def observe(
        self, simulator_state: datatypes.SimulatorState, key: jax.Array = None
    ) -> jax.Array:
        features = self._features_extractor.extract_features(simulator_state, key)
        sdc_traj_matrix, other_traj_matrix, rg_matrix, tl_matrix, gps_path_matrix = features
        sdc_traj_matrix = jnp.reshape(sdc_traj_matrix, (*simulator_state.batch_dims, -1))
        other_traj_matrix = jnp.reshape(other_traj_matrix, (*simulator_state.batch_dims, -1))
        rg_matrix = jnp.reshape(rg_matrix, (*simulator_state.batch_dims, -1))
        tl_matrix = jnp.reshape(tl_matrix, (*simulator_state.batch_dims, -1))
        gps_path_matrix = jnp.reshape(gps_path_matrix, (*simulator_state.batch_dims, -1))
        return jnp.concatenate(
            [sdc_traj_matrix, other_traj_matrix, rg_matrix, tl_matrix, gps_path_matrix],
            axis=len(simulator_state.batch_dims),
        )
