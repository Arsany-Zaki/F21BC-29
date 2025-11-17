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

	inves_details_list = expand_params(config)
	for inves_details in inves_details_list:
		printer.start_inves(inves_details)
		for group_details in inves_details.groups_details:
			for exp_detail in group_details.exps_details:
				exp_result = execute_exp(
					training_data=training_points,
					testing_data=testing_points,
					nn_config=exp_detail.nn_params,
					pso_config=exp_detail.pso_params,
					data_config=DataPrepConfig())
				exp_detail.results = exp_result

if __name__ == '__main__':
	#gc.GC_PSO_PRINT = True
	run_exp_suite()