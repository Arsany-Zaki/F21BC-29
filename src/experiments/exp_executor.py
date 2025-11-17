from nn.nn import NNParams
from pso.pso import PSOParams
from experiments.entities_yaml import *
from pso_nn_coupling.nn_trainer_with_pso import NNTrainerUsingPSO
from nn.nn import NeuralNetwork
import numpy as np
from data_prep.entities import *
from config.paths import *
from experiments.entities import *
from typing import List

def _execute_exp_run(
    training_data: List[Point], 
    testing_data: List[Point],
    nn_config: NNParams, 
    pso_config: PSOParams,
    data_config: DataPrepConfig
) -> RunResults:
    # Train neural network using PSO
    nn_trainer = NNTrainerUsingPSO(
        training_points=training_data,
        pso_config=pso_config,
        nn_config=nn_config
    )
    best_position, best_training_cost = nn_trainer.train_nn_using_pso()
    training_time_secs = 0.0  # Placeholder, update if timing is needed
    
    # Evaluate neural network on test data set (normalized values)
    nn = NeuralNetwork(config=nn_config)
    trained_nn_weights, trained_nn_biases = (
        nn_trainer.pso_vector_to_nn_weights_and_biases(np.array(best_position))
    )
    test_cost, test_predictions = (
        nn.forward_run_full_set(trained_nn_weights, trained_nn_biases, testing_data)
    )
    generalization_ratio = test_cost / best_training_cost

    # Convert neural network predictions to real (non-normalized) values
    norm_factor_mean = data_config.norm_factors[0]
    norm_factor_std = data_config.norm_factors[1]
    nn_predictions_real_vals = test_predictions * norm_factor_std + norm_factor_mean
    test_target = [p.target_real_value for p in testing_data]
    
    # Calculate neural network evaluation metrics on real (non-normalized) values
    squared_error = [(pred - target_real) ** 2 for pred, target_real in zip(nn_predictions_real_vals, test_target)]
    mse = sum(squared_error) / len(testing_data)
    rmse = mse ** 0.5
    sum_abs_error = sum(abs(pred - target_real)  for pred, target_real in zip(nn_predictions_real_vals, test_target))
    mae = sum_abs_error / len(testing_data)
    
    return RunResults(
        training_cost=best_training_cost,
        training_time_secs=training_time_secs,
        test_cost=test_cost,
        mse=mse,
        rmse=rmse,
        mae=mae,
        generalization_ratio=generalization_ratio
    )

def execute_exp(
    training_data: List[Point], 
    testing_data: List[Point],
    nn_config: NNParams, 
    pso_config: PSOParams,
    data_config: DataPrepConfig
) -> ExpResults:
    exp_result = ExpResults(
        exp_run_results=[],
        avg_training_cost=0,
        avg_training_time_secs=0,
        avg_test_cost=0,
        avg_mse=0,
        avg_rmse=0,
        avg_mae=0,
        avg_generalization_ratio=0,
        std_training_cost=0,
        std_training_time_secs=0,
        std_test_cost=0,
        std_mse=0,
        std_rmse=0,
        std_mae=0,
        std_generalization_ratio=0
    )
    exp_run_result = _execute_exp_run(
        training_data=training_data,
        testing_data=testing_data,
        nn_config=nn_config,
        pso_config=pso_config,
        data_config=data_config
    )
   
    exp_result.exp_run_results.append(exp_run_result)
    exp_result.avg_training_cost = np.mean([r.training_cost for r in exp_result.exp_run_results])
    exp_result.std_training_cost = np.std([r.training_cost for r in exp_result.exp_run_results])
    exp_result.avg_training_time_secs = np.mean([r.training_time_secs for r in exp_result.exp_run_results])
    exp_result.std_training_time_secs = np.std([r.training_time_secs for r in exp_result.exp_run_results])
    exp_result.avg_test_cost = np.mean([r.test_cost for r in exp_result.exp_run_results])
    exp_result.std_test_cost = np.std([r.test_cost for r in exp_result.exp_run_results])
    exp_result.avg_mse = np.mean([r.mse for r in exp_result.exp_run_results])
    exp_result.std_mse = np.std([r.mse for r in exp_result.exp_run_results])
    exp_result.avg_rmse = np.mean([r.rmse for r in exp_result.exp_run_results])
    exp_result.std_rmse = np.std([r.rmse for r in exp_result.exp_run_results])
    exp_result.avg_mae = np.mean([r.mae for r in exp_result.exp_run_results])
    exp_result.std_mae = np.std([r.mae for r in exp_result.exp_run_results])
    exp_result.avg_generalization_ratio = np.mean([r.generalization_ratio for r in exp_result.exp_run_results])
    exp_result.std_generalization_ratio = np.std([r.generalization_ratio for r in exp_result.exp_run_results])
    return exp_result

