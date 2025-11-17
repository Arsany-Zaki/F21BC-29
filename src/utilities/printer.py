from pso.entities import PSOParams
from nn.entities import NNParams
from experiments.entities import *
from experiments.entities_yaml import GroupConfig

class Printer:
	def __init__(self):
		pass

	def start_inves(self, metadata: dict[str, str]):
		print('**' * 4, f"Investigation: {metadata.get('name', 'Unknown')}")

	def start_group(self, exp_group: GroupConfig):
		print('**' * 3, f"Experiment Group: {exp_group.id}")
		print('  ' * 3, f"Name: {exp_group.metadata.get('name', 'Unknown')}")
		print('  ' * 3, f"Description: {exp_group.metadata.get('description', 'Unknown')}")

	def start_exp(self, metadata: dict[str, str]):
		print('**' * 2, f"Investigation: {metadata.get('name', 'Unknown')}")
	
	def inves_results_vel_coeffs(self, inves_results: InvesResults_VelCoeffs):
		print('**' * 4, f"Investigation Results Summary (Velocity Coefficients):")
		for group_result in inves_results.exp_group_results:
			self.group_results_vel_coeffs(group_result)

	def inves_results_fixed_budget(self, inves_results: InvesResults_FixedBudget):
		print('**' * 4, f"Investigation Results Summary (Fixed Budget):")
		for group_result in inves_results.exp_group_results:
			self.group_results_fixed_budget(group_result)

	def group_results_vel_coeffs(self, group_results: GroupResults_VelCoeffs):
		print('**' * 3, f"Experiment Group Results Summary:")
		for i, exp_result in enumerate(group_results.exp_results):
			print('**' * 2, f"Experiment {i + 1}:")
			self.exp_results(exp_result)

	def group_results_fixed_budget(self, group_results: GroupResults_FixedBudget):
		print('**' * 3, f"Experiment Group Results Summary:")
		for i, exp_result in enumerate(group_results.exp_results):
			print('**' * 2, f"Experiment {i + 1}:")
			self.exp_results(exp_result)

	def exp_results(self, exp_result: ExpResults):
		print('**' * 2, "Experiment Result Summary:")
		print(f"  Average Training Cost: {exp_result.avg_training_cost:.6f} ± {exp_result.std_training_cost:.6f}")
		print(f"  Average Training Time (secs): {exp_result.avg_training_time_secs:.2f} ± {exp_result.std_training_time_secs:.2f}")
		print(f"  Average Test Cost: {exp_result.avg_test_cost:.6f} ± {exp_result.std_test_cost:.6f}")
		print(f"  Average MSE: {exp_result.avg_mse:.6f} ± {exp_result.std_mse:.6f}")
		print(f"  Average RMSE: {exp_result.avg_rmse:.6f} ± {exp_result.std_rmse:.6f}")
		print(f"  Average MAE: {exp_result.avg_mae:.6f} ± {exp_result.std_mae:.6f}")
		print(f"  Average Generalization Ratio: {exp_result.avg_generalization_ratio:.6f} ± {exp_result.std_generalization_ratio:.6f}")


def pso_config_printer(pso: PSOParams):
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

def nn_config_printer(nn: NNParams):
	print("  NNConfig:")
	print(f"    input_dim: {nn.input_dim}")
	print(f"    layers_sizes: {nn.layers_sizes}")
	print(f"    activation_functions: {[a.value for a in nn.activation_functions]}")
	print(f"    cost_function: {nn.cost_function.value}")
