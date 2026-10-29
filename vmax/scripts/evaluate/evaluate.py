import argparse
from functools import partial
from pathlib import Path

import jax
import numpy as np

from . import utils


def parse_eval_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Closed-loop TRUST evaluation from scenario TFRecords."
    )
    parser.add_argument("--path_dataset", required=True)
    parser.add_argument("--path_model", required=True)
    parser.add_argument("--src_dir", default="runs")
    parser.add_argument("--eval_name", default="benchmark")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--sdc_actor", choices=["ai"], default="ai")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_num_objects", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--max_scenarios", type=int, default=None)
    parser.add_argument("--scenario_indexes", type=int, nargs="+", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--waymo_dataset", action="store_true")
    parser.add_argument("--num_paths", type=int, default=10)
    parser.add_argument("--num_points_per_path", type=int, default=300)
    args = parser.parse_args(argv)
    if (
        args.batch_size < 1
        or args.max_steps < 1
        or (args.max_scenarios is not None and args.max_scenarios < 1)
    ):
        parser.error("batch_size, max_steps and max_scenarios must be positive.")
    return args


def run_evaluation(
    env,
    data_generator,
    step_fn,
    run_path="",
    scenario_indexes=None,
    termination_keys=None,
    seed=0,
    max_steps=80,
    max_scenarios=None,
):
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")
    keys = termination_keys or ["overlap", "offroad", "run_red_light"]
    selected = None if scenario_indexes is None else set(scenario_indexes)
    rollout = jax.jit(
        jax.vmap(
            partial(
                utils.run_scenario_jit, step_fn=step_fn, reset_fn=env.reset, max_steps=max_steps
            )
        )
    )
    result = {}
    offset = 0
    count = 0
    rng = jax.random.PRNGKey(seed)
    for scenarios in data_generator:
        actual_size = scenarios.shape[0]
        indices = [
            index for index in range(actual_size) if selected is None or offset + index in selected
        ]
        scenario_ids = [offset + index for index in indices]
        offset += actual_size
        if max_scenarios is not None:
            remaining = max_scenarios - count
            indices, scenario_ids = indices[:remaining], scenario_ids[:remaining]
        if not indices:
            continue
        scenarios = jax.tree_util.tree_map(lambda x: x[np.asarray(indices)], scenarios)
        rng, batch_key = jax.random.split(rng)
        metrics, lengths = rollout(scenarios, jax.random.split(batch_key, len(indices)))
        metrics, lengths = jax.device_get((metrics, lengths))
        result = utils.append_episode_metrics(lengths, result, metrics, keys, scenario_ids)
        count += len(indices)
        if max_scenarios is not None and count >= max_scenarios:
            break
        if selected is not None and count == len(selected):
            break
    if not count:
        raise ValueError("No scenario was evaluated; check the TFRecord path and selected indices.")
    if run_path:
        utils.write_generator_result(run_path, count, result)
    return result


def main(argv=None):
    args = parse_eval_args(argv)
    from omegaconf import OmegaConf
    from vmax.simulator import datasets
    from vmax.simulator.sim_factory import make_data_generator

    run_path = Path(args.path_model).expanduser()
    if not run_path.is_dir():
        run_path = Path(args.src_dir) / args.path_model
    dataset_path = datasets.get_dataset(args.path_dataset)
    _, checkpoint_name = utils.get_model_path(run_path / "model", args.step)
    output = args.output_dir or str(
        utils.evaluation_output_path(args.eval_name, run_path, dataset_path, checkpoint_name)
    )
    env, step_fn, termination_keys = utils.setup_evaluation(
        run_path,
        output,
        args.max_num_objects,
        not args.waymo_dataset,
        args.step,
        evaluation_config={**vars(args), "path_dataset": dataset_path},
    )
    config = OmegaConf.load(run_path / ".hydra/config.yaml")
    generator = make_data_generator(
        path=dataset_path,
        max_num_objects=args.max_num_objects or config.get("max_num_objects", 64),
        include_sdc_paths=not args.waymo_dataset,
        batch_dims=(args.batch_size, 1),
        seed=None,
        repeat=1,
        drop_remainder=False,
        num_paths=args.num_paths,
        num_points_per_path=args.num_points_per_path,
    )
    result = run_evaluation(
        env,
        generator,
        step_fn,
        output,
        args.scenario_indexes,
        termination_keys,
        args.seed,
        args.max_steps,
        args.max_scenarios,
    )
    print(f"Evaluated {len(result['scenario_index'])} scenarios; results saved to {output}.")


if __name__ == "__main__":
    main()
