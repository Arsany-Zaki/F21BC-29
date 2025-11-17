import yaml
from typing import List
from dacite import from_dict
from pso.entities import PSOParams
from nn.entities import NNParams
from pso.constants import BoundHandling, InformantSelect
from nn.constants import ActFunc, CostFunc
from experiments.entities_yaml import *
from experiments.entities import *
from utilities.printer import pso_config_printer, nn_config_printer
from config.paths import * 
import itertools

def _to_enum(enum_cls, value):
	if isinstance(value, enum_cls):
		return value
	if isinstance(value, str):
		for e in enum_cls:
			if e.value == value:
				return e
	raise ValueError(f"Cannot convert {value} to {enum_cls}")

def _expand_range(val):
	if isinstance(val, list) and len(val) == 3:
		start, stop, step = val
		# inclusive range for floats
		n = int(round((stop - start) / step)) + 1
		return [round(start + i * step, 10) for i in range(n)]
	elif isinstance(val, list):
		return val
	else:
		return [val]


def _make_pso_config_grid(pso: PSOParamRanges, budget: int = None):
	inertia_range = _expand_range(pso.inertia)
	personal_range = _expand_range(pso.personal)
	global_c_range = _expand_range(pso.global_c)
	social_range = _expand_range(pso.social)
	swarm_size_range = _expand_range(pso.swarm_size)
	informants_size_range = _expand_range(pso.informants_size)
	# For fixed budget, ignore iterations_count and compute from budget
	if budget is not None:
		iter_count = [None]  # will be calculated per combo
	else:
		iter_count = [pso.iterations_count] if pso.iterations_count is not None else [None]

	combos = itertools.product(
		inertia_range, personal_range, global_c_range, social_range, swarm_size_range, informants_size_range, iter_count
	)
	for inertia, personal, global_c, social, swarm_size, informants_size, iterations_count in combos:
		if budget is not None:
			max_iter = int(budget // swarm_size)
		else:
			max_iter = iterations_count if iterations_count is not None else 0
		yield PSOParams(
			max_iter=max_iter,
			swarm_size=swarm_size,
			w_inertia=inertia,
			c_personal=personal,
			c_social=social,
			c_global=global_c,
			jump_size=pso.jump_size,
			informant_selection=_to_enum(InformantSelect, pso.informant_selection),
			informant_count=informants_size,
			boundary_handling=_to_enum(BoundHandling, pso.boundary_handling),
			dims=None,
			boundary_min=[],
			boundary_max=[],
			target_fitness=None
		)

def _make_nn_config_grid(nn: NNParamRanges):
	yield NNParams(
		input_dim=nn.input_dim,
		layers_sizes=nn.layers_sizes,
		activation_functions=[_to_enum(ActFunc, a) for a in nn.act_funcs],
		cost_function=_to_enum(CostFunc, nn.cost_func)
	)

def gen_vel_coeffs_params(all_config: Config) -> InvesParams:
	exp_groups = []
	for group in all_config.inves_vel_coeffs.groups:
		params = []
		for pso in _make_pso_config_grid(group.pso_param_ranges):
			for nn in _make_nn_config_grid(group.nn_param_ranges):
				params.append(ExpParams(pso_params=pso, nn_params=nn))
		exp_groups.append(GroupParams(exp_params=params))
	return InvesParams(exp_groups=exp_groups)

def gen_fixed_budget_params(all_config: Config) -> InvesParams:
	exp_groups = []
	for group in all_config.inves_fixed_budget.groups:
		params = []
		budget = getattr(group, 'budget', None)
		for pso in _make_pso_config_grid(group.pso_param_ranges, budget=budget):
			for nn in _make_nn_config_grid(group.nn_param_ranges):
				params.append(ExpParams(pso_params=pso, nn_params=nn))
		exp_groups.append(GroupParams(exp_params=params))
	return InvesParams(exp_groups=exp_groups)

def load_config(path: str) -> Config:
	with open(path, 'r') as f:
		config_dict = yaml.safe_load(f)
	return from_dict(data_class=Config, data=config_dict)

def float_range(x, y, z):
	vals = []
	v = x
	while v <= y + 1e-10:
		vals.append(round(v, 10))
		v += z
	return vals

def _expand_range(val):
	if isinstance(val, list) and len(val) == 3:
		return float_range(val[0], val[1], val[2])
	elif isinstance(val, list):
		return val
	else:
		return [val]


