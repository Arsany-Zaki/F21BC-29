from experiments.param_comb_generator import *
from config.paths import *
from exp_executor import *
from data_prep.data_prep import *
from data_prep.entities import *
from utilities.printer import Printer
from dacite import from_dict
import yaml
from experiments.exp_result_logger import log_config_and_result

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
	
	for inves_details in inves_details_list:
		printer.print_investigation_metadata(inves_details)
		for group_details in inves_details.groups_details:
			printer.print_group_metadata(group_details)
			printer.print_num_experiments(group_details)
			exp_total = len(group_details.exps_details)
			import time
			for idx, exp_detail in enumerate(group_details.exps_details, 1):
				printer.print_exp_progress(group_details.id, idx, exp_total, exp_detail.id)
				start_time = time.perf_counter()
				exp_result = execute_exp(
					training_data=training_points,
					testing_data=testing_points,
					nn_config=exp_detail.nn_params,
					pso_config=exp_detail.pso_params,
					analysis_config=config.analysis_config,
					data_config=DataPrepConfig())
				elapsed = time.perf_counter() - start_time
				printer.print(f"        Time taken: {elapsed:.2f} seconds")
				exp_detail.results = exp_result

	log_config_and_result(inves_details_list)

if __name__ == '__main__':
	run_exp_suite()