from pso.entities import PSOConfig
from nn.entities import NNConfig
from experiments.entities import ExpResult

class Printer:
	def __init__(self):
		pass
	def start_inves(self, metadata: dict[str, str]):
		print('** ' * 4, f"Investigation: {metadata.get('name', 'Unknown')}")
	def start_exp_group(self, metadata: dict[str, str]):
		print('** ' * 3, f"Experiment Group: {metadata['name']}")
	def start_exp(self, metadata: dict[str, str]):
		print('** ' * 2, f"Investigation: {metadata.get('name', 'Unknown')}")
	

	def exp_result(self, exp_result: ExpResult):
		print('** ' * 2, "Experiment Result Summary:")
		print(f"  Average Training Cost: {exp_result.avg_training_cost:.6f} ± {exp_result.std_training_cost:.6f}")
		print(f"  Average Training Time (secs): {exp_result.avg_training_time_secs:.2f} ± {exp_result.std_training_time_secs:.2f}")
		print(f"  Average Test Cost: {exp_result.avg_test_cost:.6f} ± {exp_result.std_test_cost:.6f}")
		print(f"  Average MSE: {exp_result.avg_mse:.6f} ± {exp_result.std_mse:.6f}")
		print(f"  Average RMSE: {exp_result.avg_rmse:.6f} ± {exp_result.std_rmse:.6f}")
		print(f"  Average MAE: {exp_result.avg_mae:.6f} ± {exp_result.std_mae:.6f}")
		print(f"  Average Generalization Ratio: {exp_result.avg_generalization_ratio:.6f} ± {exp_result.std_generalization_ratio:.6f}")


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
