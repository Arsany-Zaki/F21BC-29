from experiments.exp_params_gen import *
from config.paths import *
from exp_executor import *
from data_prep.data_prep import *
from data_prep.entities import *
import config.global_config as gc
from utilities.printer import Printer

def run_exp_suite():
	printer = Printer()
	data_prep = DataPrep(DataPrepConfig())
	training_points, testing_points = data_prep.get_normalized_input_data_split()

	config = load_config(PATH_EXP_CONFIG_FILE)
	
	vel_coeffs_inves_params = gen_vel_coeffs_params(config)
	vel_coeffs_inves_results = InvesResults_VelCoeffs(exp_group_results=[])
	printer.start_inves(config.inves_vel_coeffs.metadata)
	for group, group_params in zip(
		config.inves_vel_coeffs.groups, vel_coeffs_inves_params.exp_groups):
		printer.start_group(group)
		group_results = []
		for exp_params in group_params.exp_params:
			exp_result = execute_exp(
				training_data=training_points, 
				testing_data=testing_points,
				nn_config=exp_params.nn_params,
				pso_config=exp_params.pso_params, 
				data_config=DataPrepConfig())
			group_results.append(exp_result)
		vel_coeffs_inves_results.exp_group_results.append(
			GroupResults_VelCoeffs(exp_group_id=group.id, exp_results=group_results)
		)

	fixed_budget_inves_params = gen_fixed_budget_params(config)
	fixed_budget_inves_results = InvesResults_FixedBudget(exp_group_results=[])
	printer.start_inves(config.inves_fixed_budget.metadata)
	for group, group_params in zip(
		config.inves_fixed_budget.groups, fixed_budget_inves_params.exp_groups):
		printer.start_group(group)
		group_results = []
		for exp_params in group_params.exp_params:
			exp_result = execute_exp(
				training_data=training_points, 
				testing_data=testing_points,
				nn_config=exp_params.nn_params,
				pso_config=exp_params.pso_params,
				data_config=DataPrepConfig())
			group_results.append(exp_result)
		fixed_budget_inves_results.exp_group_results.append(
			GroupResults_FixedBudget(exp_group_id=group.id, exp_results=group_results)
		)

if __name__ == '__main__':
	#gc.GC_PSO_PRINT = True
	run_exp_suite()