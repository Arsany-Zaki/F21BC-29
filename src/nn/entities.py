from dataclasses import dataclass
from typing import List
from nn.constants import *

@dataclass
class NNParams:
    input_dim: int | None
    layers_sizes: List[int]
    activation_functions: List[ActFunc]
    cost_function: CostFunc
