# Investigation (Inves) -> Group -> Experiment (Exp) -> Run
# Run and Exp use the same set of parameters
# In a Group, some of the parameters has fixed values, others span ranges
# Inves contains multiple Groups 

from pso.entities import PSOParams
from nn.entities import NNParams
from dataclasses import dataclass

@dataclass
class RunResults:
    training_cost: float
    training_time_secs: float
    test_cost: float
    mse: float
    rmse: float
    mae: float
    generalization_ratio: float

@dataclass
class ExpResults:
    exp_run_results: list[RunResults]
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
    id: str
    pso_params: PSOParams
    nn_params: NNParams
    results: ExpResults

@dataclass
class GroupDetails:
    inves_type: str
    id: str
    metadata: dict[str, str]
    exps_details: list[ExpDetails]

@dataclass
class InvesDetails:
    inves_type: str
    id: str
    metadata: dict[str, str]
    groups_details: list[GroupDetails]




