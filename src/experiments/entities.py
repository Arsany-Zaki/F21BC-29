# Investigation (Inves) -> Group -> Experiment (Exp) -> Run
# Run and Exp use the same set of parameters
# In a Group, some of the parameters has fixed values, others span ranges
# Inves contains multiple Groups 

from pso.entities import PSOParams
from nn.entities import NNParams
from dataclasses import dataclass

@dataclass
class ExpParams:
	pso_params: PSOParams
	nn_params: NNParams

@dataclass
class GroupParams:
	exp_params: list[ExpParams]

@dataclass
class InvesParams:
    exp_groups: list[GroupParams]

@dataclass
class RunResult:
    training_cost: float
    training_time_secs: float
    test_cost: float
    mse: float
    rmse: float
    mae: float
    generalization_ratio: float

@dataclass
class ExpResults:
    exp_run_results: list[RunResult]
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

@dataclass
class ExpDetails:
    exp_params: ExpParams
    exp_results: ExpResults

@dataclass
class GroupResults_VelCoeffs:
    exp_group_id: str
    exp_results: list[ExpResults]

@dataclass
class GroupResults_FixedBudget:
    exp_group_id: str
    exp_results: list[ExpResults]

@dataclass
class InvesResults_VelCoeffs:
    exp_group_results: list[GroupResults_VelCoeffs]

@dataclass
class InvesResults_FixedBudget:
    exp_group_results: list[GroupResults_FixedBudget]