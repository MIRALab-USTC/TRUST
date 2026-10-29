from collections.abc import Iterator
from waymax import config as _config
from waymax import dataloader, datatypes, dynamics, env
from waymax import metrics as waymax_metrics
from vmax.simulator import constants, metrics, wrappers


def make_data_generator(
    path: str,
    max_num_objects: int = 128,
    include_sdc_paths: bool = True,
    seed: int | None = None,
    batch_dims: tuple = (),
    distributed: bool = False,
    repeat: int | None = None,
    drop_remainder: bool = False,
    num_paths=constants.NUM_SDC_PATHS,
    num_points_per_path=constants.NUM_POINTS_PER_SDC_PATH,
) -> Iterator[datatypes.SimulatorState]:
    return dataloader.simulator_state_generator(
        _config.DatasetConfig(
            path=path,
            max_num_objects=max_num_objects,
            batch_dims=batch_dims,
            distributed=distributed,
            shuffle_seed=seed,
            repeat=repeat,
            include_sdc_paths=include_sdc_paths,
            num_paths=num_paths,
            num_points_per_path=num_points_per_path,
            drop_remainder=drop_remainder,
        )
    )


def make_env(
    max_num_objects: int = 128,
    dynamics_model: dynamics.DynamicsModel | None = None,
    observation_type: str = "vec",
    observation_config: dict | None = None,
    reward_type: str = "linear",
    reward_config: dict = {"offroad": -1, "overlap": -1},
) -> env.PlanningAgentEnvironment:
    metrics_to_run = ["log_divergence", "overlap", "offroad"]
    metrics_to_run += ["sdc_wrongway", "sdc_progression", "sdc_off_route"]
    metrics_to_run += metrics._VMAX_METRICS_REGISTRY.keys()
    init_steps = 11
    for key in metrics._VMAX_METRICS_REGISTRY:
        if key not in waymax_metrics.get_metric_names():
            waymax_metrics.register_metric(key, metrics.get_metrics(key))
    env_config = _config.EnvironmentConfig(
        max_num_objects=max_num_objects,
        metrics=_config.MetricsConfig(metrics_to_run=metrics_to_run),
        init_steps=init_steps,
    )
    waymax_env = env.PlanningAgentEnvironment(dynamics_model, env_config)
    waymax_env = _add_observation_wrapper(waymax_env, observation_type, observation_config)
    waymax_env = _add_reward_wrapper(waymax_env, reward_type, reward_config)
    return waymax_env


def make_env_for_evaluation(
    max_num_objects: int = 128,
    dynamics_model: dynamics.DynamicsModel | None = None,
    sdc_paths_from_data: bool = True,
    observation_type: str = "vec",
    observation_config: dict | None = None,
    reward_type: str = "",
    reward_config: dict = {"offroad": -1, "overlap": -1},
    termination_keys: list[str] = ("offroad", "overlap"),
) -> env.PlanningAgentEnvironment:
    waymax_env = make_env(
        max_num_objects=max_num_objects,
        dynamics_model=dynamics_model,
        observation_type=observation_type,
        observation_config=observation_config,
        reward_type=reward_type,
        reward_config=reward_config,
    )
    waymax_env = wrappers.BraxWrapper(waymax_env, termination_keys)
    if not sdc_paths_from_data:
        waymax_env = wrappers.SDCPathWrapper(waymax_env)
    waymax_env = wrappers.VmapWrapper(waymax_env)
    return waymax_env


def _add_observation_wrapper(
    env: env.PlanningAgentEnvironment, type: str, config: dict
) -> env.PlanningAgentEnvironment:
    if config is None:
        return wrappers.ObservationWrapper(env, type)
    return wrappers.ObservationWrapper(env, type, **config)


def _add_reward_wrapper(
    env: env.PlanningAgentEnvironment, type: str, reward_config: dict
) -> env.PlanningAgentEnvironment:
    if type == "linear":
        return wrappers.RewardLinearWrapper(env, reward_config)
    if type in ("", None):
        return env
    raise ValueError(f"Unsupported reward type: {type}")
