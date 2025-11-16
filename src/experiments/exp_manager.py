from experiments.exp_params_gen import *
from config.paths import *
from exp_executor import *
from data_prep.data_prep import *
from data_prep.entities import *
import config.global_config as gc

def run_exp_suite():
	data_prep = DataPrep(DataPrepConfig())
	training_points, testing_points = data_prep.get_normalized_input_data_split()

	config = load_config(PATH_EXP_CONFIG_FILE)
	
	print("Velocity Coeffs Experiments")
	vel_coeffs_exp_params = gen_vel_coeffs_params(config)
	vel_coeffs_exp_results = []
	for exp_params in vel_coeffs_exp_params:
		exp_result = execute_exp(
			training_data=training_points, 
			testing_data=testing_points,
			nn_config=exp_params.nn_params,
			pso_config=exp_params.pso_params, 
			data_config=DataPrepConfig())
		vel_coeffs_exp_results.append(exp_result)
	
	print("Fixed Budget Experiments")
	fixed_budget_exp_params = gen_fixed_budget_params(config)
	fixed_budget_exp_results = []
	for exp_params in fixed_budget_exp_params:
		exp_result = execute_exp(
			training_data=training_points, 
		 	testing_data=testing_points,
			nn_config=exp_params.nn_params,
			pso_config=exp_params.pso_params,
			data_config=DataPrepConfig())
		fixed_budget_exp_results.append(exp_result)
	
if __name__ == '__main__':
	gc.GC_DEBUG_MODE = True
	run_exp_suite()

