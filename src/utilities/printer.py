from tabulate import tabulate
from pso.entities import PSOParams
from nn.entities import NNParams
from experiments.entities import *
from experiments.entities_yaml import GroupConfig

class Printer:
	def print_investigation_metadata(self, inves_details):
		metadata = inves_details.metadata or {}
		self.print(f"\n=== Investigation: {inves_details.id} (type: {inves_details.inves_type}) ===")
		if metadata:
			for k, v in metadata.items():
				self.print(f"  {k}: {v}")

	def print_group_metadata(self, group_details):
		metadata = group_details.metadata or {}
		self.print(f"  -- Group: {group_details.id} (type: {group_details.inves_type}) --")
		if metadata:
			for k, v in metadata.items():
				self.print(f"    {k}: {v}")

	def print_num_experiments(self, group_details):
		n = len(group_details.exps_details)
		self.print(f"    Number of experiments: {n}")

	def print_exp_progress(self, group_id, exp_idx, exp_total, exp_id):
		self.print(f"      [Group {group_id}] Running experiment {exp_idx}/{exp_total} (id: {exp_id}) ...")
	def print(self, *args, **kwargs):
		print(*args, **kwargs)
	def print_exp_params(self, exp_detail, idx):
		print("\n--- Experiment Parameters Before Execution ---")
		self.exp_details_summary(exp_detail, idx)
	def print_summary(self, inves_details_list):
		for inves_details in inves_details_list:
			metadata = inves_details.metadata or {}
			print(f"- Investigation (type: {inves_details.inves_type})")
			print(f"    id: {inves_details.id}")
			name = metadata.get('name')
			if name:
				print(f"    name: {name}")
			desc = metadata.get('description')
			if desc:
				print(f"    description: {desc}")
			for group_details in inves_details.groups_details:
				gmeta = group_details.metadata or {}
				print(f"    - Group (type: {group_details.inves_type})")
				print(f"        id: {group_details.id}")
				gname = gmeta.get('name')
				if gname:
					print(f"        name: {gname}")
				gdesc = gmeta.get('description')
				if gdesc:
					print(f"        description: {gdesc}")

	def print_full_results(self, inves_details_list):
		print("\n===== FULL EXPERIMENT RESULTS =====\n")
		for inves_details in inves_details_list:
			self.start_inves(inves_details)
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
		for i, exp_detail in enumerate(group_details.exps_details, 1):
			self.exp_details_summary(exp_detail, i)

	def exp_details_summary(self, exp_detail, idx):
		tab2 = '        '
		tab3 = '            '
		print(f"{tab2}- Experiment {idx}")
		print(f"{tab3}exp params:")
		# Print PSO params
		pso = exp_detail.pso_params
		# PSO params split into 3 tables, each with two rows (labels, values)
		pso = exp_detail.pso_params
		pso_dict = {k: v for k, v in pso.__dict__.items() if k not in ("boundary_min", "boundary_max")}
		# Merge Core and Coefficients into one table
		merged_keys = ["max_iter", "swarm_size", "jump_size", "target_fitness", "w_inertia", "c_personal", "c_social", "c_global"]
		labels = [k for k in merged_keys]
		values = [pso_dict.get(k, "-") for k in merged_keys]
		table_data = [labels, values]
		group_label = "PSO - Core & Coefficients:"
		print(f"{tab3}{group_label}")
		table_str = tabulate(table_data, tablefmt="fancy_grid", showindex=False)
		table_str = "\n".join(f"{tab3}{line}" for line in table_str.splitlines())
		print(table_str)

		# Print Other group as before
		other_keys = ["informant_selection", "informant_count", "boundary_handling"]
		labels = [k for k in other_keys]
		values = [pso_dict.get(k, "-") for k in other_keys]
		table_data = [labels, values]
		group_label = "PSO - Other:"
		print(f"{tab3}{group_label}")
		table_str = tabulate(table_data, tablefmt="fancy_grid", showindex=False)
		table_str = "\n".join(f"{tab3}{line}" for line in table_str.splitlines())
		print(table_str)

		class Printer:
			def print(self, *args, **kwargs):
				print(*args, **kwargs)
			def print_exp_params(self, exp_detail, idx):
				print("\n--- Experiment Parameters Before Execution ---")
				self.exp_details_summary(exp_detail, idx)
			def print_summary(self, inves_details_list):
				for inves_details in inves_details_list:
					metadata = inves_details.metadata or {}
					print(f"- Investigation (type: {inves_details.inves_type})")
					print(f"    id: {inves_details.id}")
					name = metadata.get('name')
					if name:
						print(f"    name: {name}")
					desc = metadata.get('description')
					if desc:
						print(f"    description: {desc}")
					for group_details in inves_details.groups_details:
						gmeta = group_details.metadata or {}
						print(f"    - Group (type: {group_details.inves_type})")
						print(f"        id: {group_details.id}")
						gname = gmeta.get('name')
						if gname:
							print(f"        name: {gname}")
						gdesc = gmeta.get('description')
						if gdesc:
							print(f"        description: {gdesc}")

			def print_full_results(self, inves_details_list):
				print("\n===== FULL EXPERIMENT RESULTS =====\n")
				for inves_details in inves_details_list:
					self.start_inves(inves_details)
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
