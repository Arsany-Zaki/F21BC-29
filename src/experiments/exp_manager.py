from experiments.param_comb_gen import *
from config.paths import *
from exp_executor import *
from data_prep.data_prep import *
from data_prep.entities import *
import config.global_config as gc
from utilities.printer import Printer
from dacite import from_dict
import yaml

def load_config(path: str) -> Config:
	with open(path, 'r') as f:
		config_dict = yaml.safe_load(f)
	return from_dict(data_class=Config, data=config_dict)

def run_exp_suite():
	printer = Printer()
	data_prep = DataPrep(DataPrepConfig())
	training_points, testing_points = data_prep.get_normalized_input_data_split()

	config = load_config(PATH_EXP_CONFIG_FILE)

	inves_details_list = expand_params(config)
	
	printer.print_summary(inves_details_list)

	for inves_details in inves_details_list:
		for group_details in inves_details.groups_details:
			for idx, exp_detail in enumerate(group_details.exps_details, 1):
				exp_result = execute_exp(
					training_data=training_points,
					testing_data=testing_points,
					nn_config=exp_detail.nn_params,
					pso_config=exp_detail.pso_params,
					data_config=DataPrepConfig())
				exp_detail.results = exp_result

	printer.print_full_results(inves_details_list)

if __name__ == '__main__':
	#gc.GC_PSO_PRINT = True
	run_exp_suite()