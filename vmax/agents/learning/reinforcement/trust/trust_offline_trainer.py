from collections.abc import Callable
import json
from pathlib import Path
import pickle
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np
from vmax.agents.learning.offline_dataset import load_offline_dataset
from vmax.agents.learning.offline_replay_buffer import OfflineReplayBuffer
from .observation import validate_trajectory_mask
from .trust_factory import initialize, make_sgd_step


def train(
    offline_dataset_path,
    total_updates: int,
    batch_size: int,
    learning_rate: float,
    network_config: dict,
    discount: float,
    tau: float,
    beta: float,
    expectile: float,
    epsilon: float,
    seed: int = 0,
    log_freq: int = 1000,
    save_freq: int = 5000,
    expected_dataset_size: int | None = 2000000,
    observation_config: dict | None = None,
    dataset_name: str = "unspecified",
    iqr_alpha: float = 3.0,
    numerical_epsilon: float = 1e-06,
    checkpoint_logdir: str = "",
    progress_fn: Callable = lambda *args: None,
):
    devices = jax.local_devices()
    num_devices = len(devices)
    if total_updates < 1 or batch_size < 1 or batch_size % num_devices:
        raise ValueError(
            "total_updates must be positive and batch_size must divide evenly across local devices."
        )
    if learning_rate <= 0 or log_freq < 0 or save_freq < 0:
        raise ValueError("learning_rate must be positive; log/save frequencies cannot be negative.")
    if jax.process_count() != 1:
        raise ValueError("This trainer supports one host with one or multiple local devices.")
    dataset = load_offline_dataset(offline_dataset_path, expected_size=expected_dataset_size)
    if observation_config is not None:
        validate_trajectory_mask(
            dataset.next_observations, dataset.trajectory_mask, observation_config
        )
    buffer = OfflineReplayBuffer(dataset, seed)
    obs_size, action_size = (dataset.observations.shape[1], dataset.actions.shape[1])
    key = jax.random.PRNGKey(seed)
    key, network_key = jax.random.split(key)
    network, state, _ = initialize(
        action_size, obs_size, learning_rate, network_config, network_key
    )
    state = jax.device_put_replicated(state, devices)
    sgd = make_sgd_step(
        network,
        discount,
        tau,
        beta,
        expectile,
        epsilon,
        iqr_alpha,
        numerical_epsilon,
        axis_name="batch",
    )
    update = jax.pmap(sgd, axis_name="batch")
    model_dir = Path(checkpoint_logdir) if checkpoint_logdir else None
    if model_dir:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "offline_dataset.json").write_text(
            json.dumps(
                {
                    "source_files": dataset.source_files,
                    "size": dataset.size,
                    "dataset_name": dataset_name,
                    "observation_size": obs_size,
                    "action_size": action_size,
                    "seed": seed,
                    "batch_size": batch_size,
                    "total_gradient_updates": total_updates,
                    "epsilon": epsilon,
                    "bootstrap_on_timeouts": True,
                },
                indent=2,
            )
        )

    def unreplicate(tree):
        return jax.tree_util.tree_map(lambda x: x[0], tree)

    def save(name):
        with (model_dir / name).open("wb") as output:
            pickle.dump(jax.device_get(unreplicate(state.params)), output)

    print(
        f"TRUST: loaded all {dataset.size:,} saved transitions; {num_devices} device(s); {total_updates:,} gradient updates.",
        flush=True,
    )
    started = perf_counter()
    for current_update in range(1, total_updates + 1):
        batch = buffer.sample(batch_size)
        batch = jax.tree_util.tree_map(
            lambda x: np.asarray(x).reshape((num_devices, batch_size // num_devices) + x.shape[1:]),
            batch,
        )
        key, step_key = jax.random.split(key)
        keys = jax.random.split(step_key, num_devices)
        (state, _), metrics = update((state, keys), batch)
        if log_freq and current_update % log_freq == 0 or current_update == total_updates:
            metrics = jax.device_get(unreplicate(metrics))
            metrics = {f"train/{name}": float(value) for name, value in metrics.items()}
            metrics["train/rl_gradient_steps"] = current_update
            metrics["train/offline_dataset_size"] = dataset.size
            progress_fn(current_update, metrics)
            print(f"[TRUST] gradient update {current_update}/{total_updates}", flush=True)
        if model_dir and save_freq and (current_update % save_freq == 0):
            save(f"model_upd_{current_update}.pkl")
    final_state = jax.device_get(unreplicate(state))
    if int(final_state.rl_gradient_steps) != total_updates:
        raise RuntimeError("Gradient-step counter does not match the requested schedule.")
    if model_dir:
        save("model_final.pkl")
    print(f"TRUST training completed in {perf_counter() - started:.2f}s", flush=True)
    return final_state
