from .abstract_extractor import AbstractFeaturesExtractor
from .utils import (
    OBJECT_MAPPING,
    RG_MAPPING,
    TL_MAPPING,
    get_feature_size,
    normalize_by_feature,
    normalize_path,
)
from .vec_extractor import VecFeaturesExtractor


def get_extractor(name):
    if name != "vec":
        raise ValueError("TRUST evaluation requires vector observations.")
    return VecFeaturesExtractor
