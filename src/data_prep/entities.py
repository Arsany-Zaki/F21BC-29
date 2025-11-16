from dataclasses import dataclass, field
from data_prep.constants import *

@dataclass
class DataPrepConfig:
    norm_method: NormMethod = NormMethod.ZSCORE
    norm_factors: list[float] = field(default_factory=lambda: [0.0, 1.0])
    split_test_size: float = 0.3    
    random_seed: int = 42

@dataclass
class Point:
    features_real_values: list[float]
    target_real_value: float
    features_norm_values: list[float]
    target_norm_value: float
