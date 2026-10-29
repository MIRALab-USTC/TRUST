import csv
import hashlib
import json
from pathlib import Path
import pickle
import re

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from vmax.agents.learning.reinforcement.trust import make_inference_fn, make_networks
from vmax.simulator.metrics.aggregators import nuplan_aggregate_score, vmax_aggregate_score
from vmax.simulator.metrics.collector import _metrics_operands


def run_scenario_jit(scenario, rng_key, step_fn, reset_fn, max_steps=80):
    rng_key, reset_key = jax.random.split(rng_key)
    reset_keys = jax.random.split(reset_key, scenario.shape[0])
    transition = reset_fn(scenario, reset_keys)
    metrics = {
        name: jnp.full((max_steps,), jnp.nan, dtype=jnp.float32) for name in transition.metrics
    }

    def condition(carry):
        state, _, _, count = carry
        return jnp.logical_and(jnp.logical_not(jnp.all(state.done)), count < max_steps)

    def body(carry):
        state, key, values, count = carry
        key, step_key = jax.random.split(key)
        state, _ = step_fn(state, key=step_key)
        values = {
            name: values[name].at[count].set(jnp.asarray(value).reshape(-1)[0])
            for name, value in state.metrics.items()
        }
        return state, key, values, count + 1

    _, _, metrics, count = jax.lax.while_loop(
        condition, body, (transition, rng_key, metrics, jnp.array(0, dtype=jnp.int32))
    )
    return metrics, count


def get_model_path(model_path, step=None):
    path = Path(model_path).expanduser()
    if path.is_file() and step is None:
        return str(path), path.name
    if step is not None:
        selected = path / f"model_upd_{step}.pkl"
    elif (path / "model_final.pkl").is_file():
        selected = path / "model_final.pkl"
    else:
        candidates = [
            (int(match.group(1)), p)
            for p in path.glob("model_upd_*.pkl")
            if (match := re.fullmatch(r"model_upd_(\d+)\.pkl", p.name))
        ]
        if not candidates:
            raise FileNotFoundError(f"No TRUST checkpoint found in {path}.")
        selected = max(candidates, key=lambda pair: pair[0])[1]
    if not selected.is_file():
        raise FileNotFoundError(selected)
    return str(selected), selected.name


def evaluation_output_path(root, run_path, dataset_path, checkpoint_name):
    identity = str(dataset_path)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:10]
    dataset_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", Path(identity).name) or "dataset"
    return (
        Path(root) / Path(run_path).name / f"{dataset_name}_{digest}" / Path(checkpoint_name).stem
    )


def load_policy(model_path, network_config, observation_size, action_size, learning_rate=1e-4):
    network = make_networks(
        observation_size, action_size, learning_rate=learning_rate, network_config=network_config
    )
    with Path(model_path).open("rb") as handle:
        params = pickle.load(handle)
    return make_inference_fn(network)(params.policy, deterministic=True)


def setup_evaluation(
    run_path,
    output_path,
    max_num_objects=None,
    sdc_paths_from_data=True,
    step=None,
    evaluation_config=None,
):
    from waymax import dynamics
    from vmax.agents.pipeline.inference import policy_step
    from vmax.simulator.sim_factory import make_env_for_evaluation
    from functools import partial

    run_path = Path(run_path)
    config = OmegaConf.to_container(OmegaConf.load(run_path / ".hydra/config.yaml"), resolve=True)
    if config["algorithm"]["name"].upper() != "TRUST":
        raise ValueError("This evaluation entrypoint only loads TRUST checkpoints.")
    termination_keys = config.get("termination_keys", ["overlap", "offroad", "run_red_light"])
    reward_config = config.get(
        "reward_config",
        {
            "overlap": -1.0,
            "offroad": -1.0,
            "red_light": -1.0,
            "off_route": -0.6,
            "progression": 0.2,
        },
    )
    env = make_env_for_evaluation(
        max_num_objects=max_num_objects or config.get("max_num_objects", 64),
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=sdc_paths_from_data,
        observation_type=config["observation_type"],
        observation_config=config["observation_config"],
        reward_type="linear",
        reward_config=reward_config,
        termination_keys=termination_keys,
    )
    model_path, model_name = get_model_path(run_path / "model", step)
    network_config = dict(config["network"])
    network_config["state_value"] = config["algorithm"]["network"]["state_value"]
    policy = load_policy(
        model_path,
        network_config,
        env.observation_spec(),
        env.action_spec().data.shape[0],
        config["algorithm"]["learning_rate"],
    )
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "evaluation_config.json").write_text(
        json.dumps(
            {
                "checkpoint": model_path,
                "checkpoint_name": model_name,
                "termination_keys": termination_keys,
                "reward_config": reward_config,
                "observation_config": config["observation_config"],
                "sdc_paths_from_data": sdc_paths_from_data,
                "evaluation": evaluation_config or {},
            },
            indent=2,
        )
    )
    return env, partial(policy_step, env=env, policy_fn=policy), termination_keys


def append_episode_metrics(steps_done, eval_metrics, episode_metrics, term_keys, scenario_indices):
    for index, (steps, scenario_index) in enumerate(
        zip(np.asarray(steps_done).reshape(-1), scenario_indices)
    ):
        steps = int(steps)
        if steps <= 0:
            raise ValueError(
                f"Scenario {scenario_index} finished before any valid evaluation step."
            )
        row = {"scenario_index": int(scenario_index), "episode_length": steps}
        for name, values in episode_metrics.items():
            values = np.asarray(values)[index, :steps]
            operation = _metrics_operands.get(name, np.mean)
            if isinstance(operation, dict):
                row.update({key: float(function(values)) for key, function in operation.items()})
            else:
                row[name] = float(operation(values))
        row["accuracy"] = float(not any(row.get(key, 0) > 0 for key in term_keys))
        nuplan_keys = {
            "ttc_within_bound",
            "progress_ratio",
            "nuplan_speed_compliance",
            "comfort",
            "at_fault_collision",
            "offroad",
            "making_progress",
            "nuplan_driving_direction_compliance",
        }
        if nuplan_keys <= row.keys():
            row["nuplan_aggregate_score"] = float(nuplan_aggregate_score(row))
        if nuplan_keys | {"run_red_light", "multiple_lanes_score"} <= row.keys():
            row["vmax_aggregate_score"] = float(vmax_aggregate_score(row))
        for key, value in row.items():
            eval_metrics.setdefault(key, []).append(value)
    return eval_metrics


def write_generator_result(save_path, num_scenarios, metrics):
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    names = list(metrics)
    with (path / "evaluation_episodes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerows(zip(*(metrics[name] for name in names)))
    summary = {}
    for name, values in metrics.items():
        if name != "scenario_index" and values:
            value = float(np.mean(values))
            summary[name] = value if np.isfinite(value) else None
    (path / "evaluation_results.json").write_text(
        json.dumps({"num_scenarios": num_scenarios, "metrics": summary}, indent=2, allow_nan=False)
    )
    lines = [f"Evaluation - {num_scenarios} scenarios"]
    lines += [
        f"{name}: {value:.6f}" if value is not None else f"{name}: unavailable"
        for name, value in summary.items()
    ]
    (path / "evaluation_results.txt").write_text("\n".join(lines) + "\n")
