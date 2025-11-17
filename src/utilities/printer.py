from pso.entities import PSOParams
from nn.entities import NNParams
from experiments.entities import *
from experiments.entities_yaml import GroupConfig

class Printer:
	def __init__(self):
		pass

	def start_inves(self, inves_details: InvesDetails):
		metadata = inves_details.metadata or {}
		tab1 = '    '
		tab2 = '        '
		print(f"- Investigation (type: {inves_details.inves_type})")
		print(f"{tab1}id: {inves_details.id}")
		name = metadata.get('name')
		if name:
			print(f"{tab1}name: {name}")
		desc = metadata.get('description')
		if desc:
			print(f"{tab1}description: {desc}")
		for group_details in inves_details.groups_details:
			self.group_summary(group_details)
		print()

	def group_summary(self, group_details: GroupDetails):
		metadata = group_details.metadata or {}
		tab1 = '    '
		tab2 = '        '
		print(f"{tab1}- Group (type: {group_details.inves_type})")
		print(f"{tab2}id: {group_details.id}")
		name = metadata.get('name')
		if name:
			print(f"{tab2}name: {name}")
		desc = metadata.get('description')
		if desc:
			print(f"{tab2}description: {desc}")

	def start_exp(self, exp_details: ExpDetails):
		print('**' * 2, f"Experiment ID: {exp_details.id}")
		print('    ', f"PSO Params: {exp_details.pso_params}")
		print('    ', f"NN Params: {exp_details.nn_params}")
	
	def inves_details(self, inves_details: InvesDetails):
		self.start_inves(inves_details)
		for group_details in inves_details.groups_details:
			self.group_details(group_details)

	def group_details(self, group_details: GroupDetails):
		self.start_group(group_details)
		for i, exp_detail in enumerate(group_details.exps_details):
			self.start_exp(exp_detail)
			self.exp_details(exp_detail)

	def exp_details(self, exp_details: ExpDetails):
		if exp_details.results is None:
			print('      No results available.')
			return
		print('      Experiment Result Summary:')
		print(f"        - Average Training Cost: {exp_details.results.avg_training_cost:.6f} ± {exp_details.results.std_training_cost:.6f}")
		print(f"        - Average Training Time (secs): {exp_details.results.avg_training_time_secs:.2f} ± {exp_details.results.std_training_time_secs:.2f}")
		print(f"        - Average Test Cost: {exp_details.results.avg_test_cost:.6f} ± {exp_details.results.std_test_cost:.6f}")
		print(f"        - Average MSE: {exp_details.results.avg_mse:.6f} ± {exp_details.results.std_mse:.6f}")
		print(f"        - Average RMSE: {exp_details.results.avg_rmse:.6f} ± {exp_details.results.std_rmse:.6f}")
		print(f"        - Average MAE: {exp_details.results.avg_mae:.6f} ± {exp_details.results.std_mae:.6f}")
		print(f"        - Average Generalization Ratio: {exp_details.results.avg_generalization_ratio:.6f} ± {exp_details.results.std_generalization_ratio:.6f}")

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
