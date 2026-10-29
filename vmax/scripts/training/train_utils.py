import logging
from datetime import datetime
import jax
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def resolve_output_dir(
    algorithm_name: str, observation_type: str, encoder_config: dict, name_run: str, name_exp: str
) -> str:
    _name_exp = "runs" if name_exp is None else name_exp
    if name_run is None:
        _name_run = f"{algorithm_name.upper()}_{observation_type.upper()}"
        encoder_type = encoder_config["type"]
        if encoder_type != "none":
            _name_run += f"_{encoder_type.upper()}"
        _name_run += f"_{datetime.now().strftime('%d-%m_%H:%M:%S')}/"
    else:
        _name_run = name_run
    return f"{_name_exp}/{_name_run}"


def log_metrics(num_steps: int, metrics: dict, writer: SummaryWriter = None) -> None:
    for key, value in metrics.items():
        if writer:
            writer.add_scalar(key, value, num_steps)
        logger.info(f"{key}: {value}")


def print_hyperparameters(args: dict) -> None:
    print(" Experiment Summary ".center(40, "="))
    print(f"- Algorithm          : {args['algorithm']['name']}")
    print(f"- Observation Type   : {args['observation_type']}")
    path = args.get("offline_dataset_path")
    print(f"- Dataset Path       : {path}")
    print(f"- Total Timesteps    : {args['total_timesteps']}")


def get_and_print_device_info() -> int:
    print(" Devices ".center(40, "="))
    print(f"- Backend: {jax.default_backend()}")
    print(f"- Devices: {jax.local_device_count()} -> {jax.local_devices()}")


def setup_tensorboard(run_path: str) -> SummaryWriter:
    return SummaryWriter(log_dir=run_path)
