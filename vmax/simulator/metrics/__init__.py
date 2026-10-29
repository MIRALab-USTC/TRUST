import importlib

_CLASS_MODULES = {
    "AtFaultCollisionMetric": "at_fault_collision",
    "ComfortMetric": "comfort",
    "DrivingDirectionComplianceMetric": "driving_direction_compliance",
    "OnMultipleLanesMetric": "on_multiple_lanes",
    "ProgressRatioMetric": "progress_ratio",
    "RunRedLightMetric": "red_light",
    "OffRouteMetric": "route",
    "SpeedLimitViolationMetric": "speed_limit",
    "TimeToCollisionMetric": "ttc",
}
_REGISTRY_NAMES = {
    "run_red_light": "RunRedLightMetric",
    "ttc": "TimeToCollisionMetric",
    "at_fault_collision": "AtFaultCollisionMetric",
    "comfort": "ComfortMetric",
    "speed_limit": "SpeedLimitViolationMetric",
    "on_multiple_lanes": "OnMultipleLanesMetric",
    "driving_direction_compliance": "DrivingDirectionComplianceMetric",
    "progress_ratio_nuplan": "ProgressRatioMetric",
}
_REGISTRY = None


def __getattr__(name):
    if name == "_VMAX_METRICS_REGISTRY":
        return _registry()
    if name in _CLASS_MODULES:
        module = importlib.import_module(f"{__name__}.{_CLASS_MODULES[name]}")
        return getattr(module, name)
    if name == "infer_speed_limit_from_simulator_state":
        return importlib.import_module(
            f"{__name__}.speed_limit"
        ).infer_speed_limit_from_simulator_state
    raise AttributeError(name)


def _registry():
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = {key: __getattr__(value)() for key, value in _REGISTRY_NAMES.items()}
    return _REGISTRY


def get_metrics(name):
    return _registry()[name]
