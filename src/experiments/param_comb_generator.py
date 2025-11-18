from typing import List
from pso.entities import PSOParams
from nn.entities import NNParams
from experiments.entities import ExpDetails, InvesDetails, GroupDetails
from pso.constants import BoundHandling, InformantSelect
from nn.constants import ActFunc, CostFunc
from experiments.entities_yaml import Config
from config.paths import *
from experiments.constants import InvesType

def _expand_range(param_range):
    import numpy as np
    if isinstance(param_range, (list, tuple)):
        if len(param_range) == 3 and all(isinstance(x, (int, float)) for x in param_range):
            start, end, step = param_range
            if start == end:
                return [start]
            else:
                # Use np.arange with rounding to avoid floating point issues
                arr = np.arange(start, end + step/2, step)
                return [float(round(x, 8)) for x in arr]
        else:
            return list(param_range)
    else:
        return [param_range]

def _make_nn_config_grid(nn_ranges):
    activation_functions = [ActFunc(a) if not isinstance(a, ActFunc) else a for a in getattr(nn_ranges, 'act_funcs', [])]
    cost_function = CostFunc(getattr(nn_ranges, 'cost_func', None)) if getattr(nn_ranges, 'cost_func', None) is not None else None
    return [NNParams(
        input_dim=nn_ranges.input_dim,
        layers_sizes=nn_ranges.layers_sizes,
        activation_functions=activation_functions,
        cost_function=cost_function
    )]

def _to_enum(enum_class, value):
    return enum_class(value)

def expand_params(all_config: Config) -> List[InvesDetails]:
    if not hasattr(all_config, 'investigations') or not all_config.investigations:
        return []
    investigations_details = []
    for inves in all_config.investigations:
        inves_type = InvesType(inves.type)
        groups_details = []
        for group in inves.groups:
            exps_details = []
            pso_ranges = group.pso_param_ranges
            nn_ranges = group.nn_param_ranges
            if inves_type == InvesType.VEL_COEFFS:
                exps_details.extend(_expand_vel_coeffs(group, pso_ranges, nn_ranges))
            elif inves_type == InvesType.FIXED_BUDGET:
                exps_details.extend(_expand_fixed_budget(group, pso_ranges, nn_ranges))
            else:
                raise ValueError(f"Unknown investigation type: {inves_type}")
            groups_details.append(GroupDetails(
                inves_type=inves_type,
                id=group.id,
                metadata=group.metadata,
                exps_details=exps_details
            ))
        investigations_details.append(InvesDetails(
            inves_type=inves_type,
            id=inves.id,
            metadata=inves.metadata,
            groups_details=groups_details
        ))
    return investigations_details

def _expand_vel_coeffs(group, pso_ranges, nn_ranges):
    exps_details = []
    inertia_range = _expand_range(pso_ranges.inertia)
    personal_range = _expand_range(pso_ranges.personal)
    global_c_range = _expand_range(pso_ranges.global_c)
    social_range = _expand_range(pso_ranges.social)
    swarm_size = int(_expand_range(pso_ranges.swarm_size)[0])
    informants_size = int(_expand_range(pso_ranges.informants_size)[0])
    iter_count = pso_ranges.iterations_count if pso_ranges.iterations_count is not None else 0
    dims = getattr(nn_ranges, 'input_dim', 1) or 1
    if not isinstance(dims, int) or dims < 1:
        raise ValueError(f"Invalid dims for PSO: {dims}. Must be a positive integer.")
    boundary_min = [0.0] * dims
    boundary_max = [1.0] * dims
    if not (isinstance(boundary_min, list) and isinstance(boundary_max, list)):
        raise ValueError(f"boundary_min and boundary_max must be lists, got {type(boundary_min)}, {type(boundary_max)}")
    if len(boundary_min) != dims or len(boundary_max) != dims:
        raise ValueError(f"boundary_min and boundary_max must have length equal to dims ({dims}), got {len(boundary_min)}, {len(boundary_max)}")
    for inertia in inertia_range:
        for personal in personal_range:
            for global_c in global_c_range:
                for social in social_range:
                    pso = PSOParams(
                        max_iter=int(iter_count),
                        swarm_size=swarm_size,
                        w_inertia=inertia,
                        c_personal=personal,
                        c_social=social,
                        c_global=global_c,
                        jump_size=pso_ranges.jump_size,
                        informant_selection=_to_enum(InformantSelect, pso_ranges.informant_selection),
                        informant_count=informants_size,
                        boundary_handling=_to_enum(BoundHandling, pso_ranges.boundary_handling),
                        dims=int(dims),
                        boundary_min=boundary_min,
                        boundary_max=boundary_max,
                        target_fitness=None
                    )
                    for nn in _make_nn_config_grid(nn_ranges):
                        exp_id = f"{group.id}_pso_{hash(str(pso))}_nn_{hash(str(nn))}"
                        exps_details.append(ExpDetails(
                            id=exp_id,
                            pso_params=pso,
                            nn_params=nn,
                            results=None
                        ))
    return exps_details

def _expand_fixed_budget(group, pso_ranges, nn_ranges):
    exps_details = []
    inertia = _expand_range(pso_ranges.inertia)[0]
    personal = _expand_range(pso_ranges.personal)[0]
    global_c = _expand_range(pso_ranges.global_c)[0]
    social = _expand_range(pso_ranges.social)[0]
    swarm_size_range = _expand_range(pso_ranges.swarm_size)
    informants_size_range = _expand_range(pso_ranges.informants_size)
    budget = getattr(group, 'budget', None)
    dims = getattr(nn_ranges, 'input_dim', 1) or 1
    if not isinstance(dims, int) or dims < 1:
        raise ValueError(f"Invalid dims for PSO: {dims}. Must be a positive integer.")
    boundary_min = [0.0] * dims
    boundary_max = [1.0] * dims
    if not (isinstance(boundary_min, list) and isinstance(boundary_max, list)):
        raise ValueError(f"boundary_min and boundary_max must be lists, got {type(boundary_min)}, {type(boundary_max)}")
    if len(boundary_min) != dims or len(boundary_max) != dims:
        raise ValueError(f"boundary_min and boundary_max must have length equal to dims ({dims}), got {len(boundary_min)}, {len(boundary_max)}")
    # Calculate (swarm_size, max_iter) pairs
    swarm_iter_pairs = [(int(swarm_size), int(budget // swarm_size) if budget is not None else 0)
                       for swarm_size in swarm_size_range]
    # Pair each with the corresponding informant_size (by index)
    n = min(len(swarm_iter_pairs), len(informants_size_range))
    for i in range(n):
        swarm_size, max_iter = swarm_iter_pairs[i]
        informants_size = int(informants_size_range[i])
        pso = PSOParams(
            max_iter=int(max_iter),
            swarm_size=swarm_size,
            w_inertia=inertia,
            c_personal=personal,
            c_social=social,
            c_global=global_c,
            jump_size=pso_ranges.jump_size,
            informant_selection=_to_enum(InformantSelect, pso_ranges.informant_selection),
            informant_count=informants_size,
            boundary_handling=_to_enum(BoundHandling, pso_ranges.boundary_handling),
            dims=int(dims),
            boundary_min=boundary_min,
            boundary_max=boundary_max,
            target_fitness=None
        )
        for nn in _make_nn_config_grid(nn_ranges):
            exp_id = f"{group.id}_pso_{hash(str(pso))}_nn_{hash(str(nn))}"
            exps_details.append(ExpDetails(
                id=exp_id,
                pso_params=pso,
                nn_params=nn,
                results=None
            ))
    return exps_details



