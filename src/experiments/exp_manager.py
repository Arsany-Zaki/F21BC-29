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
	
	vel_coeffs_gen_params = gen_vel_coeffs_params(config)
	vel_coeffs_exp_results = []
	printer.start_inves(config.inves_vel_coeffs.metadata)
	for group_config, gen_params in zip(
		config.inves_vel_coeffs.exp_groups, vel_coeffs_gen_params):
		exp_group = list(group_config.values())[0]
		printer.start_exp_group(exp_group)
		for exp_params in gen_params.exp_params:
			exp_result = execute_exp(
				training_data=training_points, 
				testing_data=testing_points,
				nn_config=exp_params.nn_params,
				pso_config=exp_params.pso_params, 
				data_config=DataPrepConfig())
			vel_coeffs_exp_results.append(exp_result)
	
	
	fixed_budget_gen_params = gen_fixed_budget_params(config)
	fixed_budget_exp_results = []
	printer.start_inves(config.inves_fixed_budget.metadata)
	for group_config, gen_params in zip(
		config.inves_fixed_budget.exp_groups, fixed_budget_gen_params):
		exp_group = list(group_config.values())[0]
		printer.start_exp_group(exp_group)
		for exp_params in gen_params.exp_params:
			exp_result = execute_exp(
				training_data=training_points, 
				testing_data=testing_points,
				nn_config=exp_params.nn_params,
				pso_config=exp_params.pso_params,
				data_config=DataPrepConfig())
			fixed_budget_exp_results.append(exp_result)
	
if __name__ == '__main__':
	#gc.GC_PSO_PRINT = True
	run_exp_suite()

