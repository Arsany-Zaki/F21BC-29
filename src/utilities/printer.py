from pso.entities import PSOConfig
from nn.entities import NNConfig

def pso_config_printer(pso: PSOConfig):
	print("  PSOConfig:")
	print(f"    max_iter: {pso.max_iter}")
	print(f"    swarm_size: {pso.swarm_size}")
	print(f"    w_inertia: {pso.w_inertia}")
	print(f"    c_personal: {pso.c_personal}")
	print(f"    c_social: {pso.c_social}")
	print(f"    c_global: {pso.c_global}")
	print(f"    jump_size: {pso.jump_size}")
	print(f"    informant_selection: {pso.informant_selection}")
	print(f"    informant_count: {pso.informant_count}")
	print(f"    boundary_handling: {pso.boundary_handling}")
	print(f"    dims: {pso.dims}")
	print(f"    boundary_min: {pso.boundary_min}")
	print(f"    boundary_max: {pso.boundary_max}")
	print(f"    target_fitness: {pso.target_fitness}")

def nn_config_printer(nn: NNConfig):
	print("  NNConfig:")
	print(f"    input_dim: {nn.input_dim}")
	print(f"    layers_sizes: {nn.layers_sizes}")
	print(f"    activation_functions: {[a.value for a in nn.activation_functions]}")
	print(f"    cost_function: {nn.cost_function.value}")
