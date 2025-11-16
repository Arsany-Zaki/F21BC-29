from nn.nn import NNConfig
from pso.pso import PSOConfig
from experiments.entities_yaml import *
from pso_nn_coupling.nn_trainer_with_pso import NNTrainerUsingPSO
from nn.nn import NeuralNetwork
import numpy as np
from data_prep.entities import *
from config.paths import *

class ExpRun:
    def __init__(
        self,
        nn_config: NNConfig,
        pso_config: PSOConfig,
        data_config: DataPrepConfig,
        training_data: list[Point],
        testing_data: list[Point],
        normalization_factors=None
    ):
        self.nn_config: NNConfig = nn_config
        self.pso_config: PSOConfig = pso_config
        self.data_config: DataPrepConfig = data_config
        self.training_data: list[Point] = training_data
        self.testing_data: list[Point] = testing_data
        self.normalization_factors = normalization_factors

    def Execute(self) -> ExpRunResult:
        
        # Train neural network using PSO
        nn_trainer = NNTrainerUsingPSO(
            training_points=self.training_data,
            pso_config=self.pso_config,
            nn_config=self.nn_config
        )
        best_position, best_training_cost = nn_trainer.train_nn_using_pso()
        training_time_secs = 0.0  # Placeholder, update if timing is needed
        
        # Evaluate neural network on test data set (normalized values)
        nn = NeuralNetwork(config=self.nn_config)
        trained_nn_weights, trained_nn_biases = (
            nn_trainer.pso_vector_to_nn_weights_and_biases(np.array(best_position))
        )
        test_cost, test_predictions = (
            nn.forward_run_full_set(trained_nn_weights, trained_nn_biases, self.testing_data)
        )
        generalization_ratio = test_cost / best_training_cost

        # Convert neural network predictions to real (non-normalized) values
        norm_factor_mean = self.normalization_factors[0][-1] if self.normalization_factors else 0.0
        norm_factor_std = self.normalization_factors[1][-1] if self.normalization_factors else 1.0
        nn_predictions_real_vals = test_predictions * norm_factor_std + norm_factor_mean
        test_target = [p.target_real_value for p in self.testing_data]
        
        # Calculate neural network evaluation metrics on real (non-normalized) values
        squared_error = [(pred - target_real) ** 2 for pred, target_real in zip(nn_predictions_real_vals, test_target)]
        mse = sum(squared_error) / len(self.testing_data)
        rmse = mse ** 0.5
        mae = sum(abs(pred - target_real) for pred, target_real in zip(nn_predictions_real_vals, test_target)) / len(self.testing_data)
        
        return ExpRunResult(
            training_cost=best_training_cost,
            training_time_secs=training_time_secs,
            test_cost=test_cost,
            mse=mse,
            rmse=rmse,
            mae=mae,
            generalization_ratio=generalization_ratio
        )

class Exp:
    def __init__(
        self, 
        nn_config: NNConfig, 
        pso_config: PSOConfig,
        data_config: DataPrepConfig,
        training_data,
        testing_data,
        normalization_factors=None
    ) -> None:
        self.nn_config: NNConfig = nn_config
        self.pso_config: PSOConfig = pso_config
        self.data_config: DataPrepConfig = data_config
        self.training_data = training_data
        self.testing_data = testing_data
        self.normalization_factors = normalization_factors

    def execute(self) -> list[ExpRunResult]:
        result = ExpResult(
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
        exp_run = ExpRun(
            nn_config=self.nn_config,
            pso_config=self.pso_config,
            data_config=self.data_config,
            training_data=self.training_data,
            testing_data=self.testing_data,
            normalization_factors=self.normalization_factors
        )
        run_result = exp_run.Execute()
        result.exp_run_results.append(run_result)
        result.avg_training_cost = np.mean([r.training_cost for r in result.exp_run_results])
        result.std_training_cost = np.std([r.training_cost for r in result.exp_run_results])
        result.avg_training_time_secs = np.mean([r.training_time_secs for r in result.exp_run_results])
        result.std_training_time_secs = np.std([r.training_time_secs for r in result.exp_run_results])
        result.avg_test_cost = np.mean([r.test_cost for r in result.exp_run_results])
        result.std_test_cost = np.std([r.test_cost for r in result.exp_run_results])
        result.avg_mse = np.mean([r.mse for r in result.exp_run_results])
        result.std_mse = np.std([r.mse for r in result.exp_run_results])
        result.avg_rmse = np.mean([r.rmse for r in result.exp_run_results])
        result.std_rmse = np.std([r.rmse for r in result.exp_run_results])
        result.avg_mae = np.mean([r.mae for r in result.exp_run_results])
        result.std_mae = np.std([r.mae for r in result.exp_run_results])
        result.avg_generalization_ratio = np.mean([r.generalization_ratio for r in result.exp_run_results])
        result.std_generalization_ratio = np.std([r.generalization_ratio for r in result.exp_run_results])
        return result

