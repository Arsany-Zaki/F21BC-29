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
	metadata: Dict[str, Any]
	pso_param_ranges: PSOParamRanges
	nn_param_ranges: NNParamRanges
	budget: Optional[int] = None

@dataclass
class InvesConfig:
	metadata: Dict[str, Any]
	groups: List[GroupConfig]

@dataclass
class AnalysisConfig:
	runs_count: int

@dataclass
class Config:
	analysis_config: AnalysisConfig
	inves_vel_coeffs: InvesConfig
	inves_fixed_budget: InvesConfig


