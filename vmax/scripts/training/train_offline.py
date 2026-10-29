import os
from functools import partial
import hydra
from omegaconf import DictConfig, OmegaConf
from vmax import PATH_TO_APP
from vmax.scripts.training import train_utils

OmegaConf.register_new_resolver("output_dir", train_utils.resolve_output_dir, replace=True)


def trust_arguments(config: dict) -> dict:
    if not config.get("offline_dataset_path"):
        raise ValueError(
            "TRUST requires offline_dataset_path pointing to saved .npz transitions; no expert fallback exists."
        )
    algorithm = config["algorithm"]
    network = dict(config["network"])
    network["state_value"] = dict(algorithm["network"]["state_value"])
    return dict(
        offline_dataset_path=config["offline_dataset_path"],
        total_updates=config["total_timesteps"],
        batch_size=algorithm["batch_size"],
        learning_rate=algorithm["learning_rate"],
        network_config=network,
        discount=algorithm["discount"],
        tau=algorithm["tau"],
        beta=algorithm["beta"],
        expectile=algorithm["expectile"],
        epsilon=algorithm["epsilon"],
        seed=config["seed"],
        log_freq=algorithm["log_freq"],
        save_freq=config["save_freq"],
        expected_dataset_size=algorithm["expected_dataset_size"],
        observation_config=config["observation_config"],
        dataset_name=config["dataset"]["name"],
        iqr_alpha=algorithm["iqr_alpha"],
        numerical_epsilon=algorithm["numerical_epsilon"],
    )


def run_trust(config: dict) -> None:
    from vmax.agents.learning.reinforcement.trust.trust_offline_trainer import train

    kwargs = trust_arguments(config)
    run_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    writer = train_utils.setup_tensorboard(run_path)
    try:
        train(
            **kwargs,
            checkpoint_logdir=os.path.join(run_path, "model"),
            progress_fn=partial(train_utils.log_metrics, writer=writer)
        )
    finally:
        writer.close()


@hydra.main(version_base=None, config_name="base_config", config_path=PATH_TO_APP + "/config")
def run(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    train_utils.print_hyperparameters(config)
    train_utils.get_and_print_device_info()
    run_trust(config)


if __name__ == "__main__":
    run()
