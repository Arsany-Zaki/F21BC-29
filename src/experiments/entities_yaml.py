# data classes for experiment suite configuration experiments/config.yaml

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class PSOParamRanges:
	inertia: Any
	personal: Any
	global_c: Any
	social: Any
	swarm_size: Any
	informants_size: Any
	informant_selection: str
	boundary_handling: str
	jump_size: float
	iterations_count: Optional[int] = None

@dataclass
class NNParamRanges:
	input_dim: int
	layers_sizes: List[int]
	act_funcs: List[str]
	cost_func: str

@dataclass
class GroupConfig:
	id: str
	pso_param_ranges: PSOParamRanges
	nn_param_ranges: NNParamRanges
	budget: Optional[int] = None
	metadata: Dict[str, Any] = None

@dataclass
class InvesConfig:
	type: str
	id: str
	groups: List[GroupConfig]
	metadata: Dict[str, Any] = None

@dataclass
class AnalysisConfig:
	runs_count: int

@dataclass
class Config:
	analysis_config: AnalysisConfig
	investigations: List[InvesConfig]



