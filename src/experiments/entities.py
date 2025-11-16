from pso.entities import PSOConfig
from nn.entities import NNConfig
from dataclasses import dataclass

@dataclass
class ExpParams:
	pso_params: PSOConfig
	nn_params: NNConfig
