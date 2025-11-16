from pso.entities import PSOConfig
from nn.entities import NNConfig
from dataclasses import dataclass

@dataclass
class ExpParams:
	pso_params: PSOConfig
	nn_params: NNConfig

@dataclass
class ExpRunResult:
    training_cost: float
    training_time_secs: float
    test_cost: float
    mse: float
    rmse: float
    mae: float
    generalization_ratio: float

@dataclass
class ExpResult:
    exp_run_results: list[ExpRunResult]
    avg_training_cost: float
    avg_training_time_secs: float
    avg_test_cost: float
    avg_mse: float
    avg_rmse: float
    avg_mae: float
    avg_generalization_ratio: float
    std_training_cost: float
    std_training_time_secs: float
    std_test_cost: float
    std_mse: float
    std_rmse: float
    std_mae: float
    std_generalization_ratio: float