from nn.nn import NeuralNetwork
import numpy as np
from nn import *
from pso.pso import PSO
from nn.constants import *
from data_prep.input_data_models import Point
from typing import List

class NNTrainerUsingPSO:
	def __init__(self, training_points: List[Point], pso_config, nn_config):
		self.training_points = training_points
		self.pso_config = pso_config
		self.nn_config = nn_config
		self.nn = None
		self.pso = None

	def train_nn_using_pso(self):
		# Create neural network
		self.nn = NeuralNetwork(config=self.nn_config)
		# Set PSO boundaries based on NN topology and activations
		boundaries = self.calculate_pso_feature_boundaries()
		self.pso_config.dims = len(boundaries)
		self.pso_config.boundary_min = [b[0] for b in boundaries]
		self.pso_config.boundary_max = [b[1] for b in boundaries]
		# Create PSO
		self.pso = PSO(self.pso_config)
		# Run PSO optimize
		best_position, best_fitness = self.pso.optimize(self._assess_fitness)
		return best_position, best_fitness

	def _assess_fitness(self, flat_weights_and_biases: np.ndarray) -> float:
		weights_struct, biases_struct = self.pso_vector_to_nn_weights_and_biases(flat_weights_and_biases)
		cost, _ = self.nn.forward_run_full_set(
			weights=weights_struct,
			biases=biases_struct,
			training_points=self.training_points
		)
		return cost

	def pso_vector_to_nn_weights_and_biases(
			self, flat_vector: np.ndarray) -> tuple[list[list[list[float]]], list[list[float]]]:
		layer_sizes = self.nn_config.layers_sizes
		input_dim = self.nn_config.input_dim
		full_layer_sizes = [input_dim] + layer_sizes
		weights_struct = []
		biases_struct = []
		idx = 0
		for l in range(1, len(full_layer_sizes)):
			n_inputs = full_layer_sizes[l-1]
			n_neurons = full_layer_sizes[l]
			# Input weights: for each input, all neurons (input-major order)
			input_weights = []  # shape: [n_inputs][n_neurons]
			for i in range(n_inputs):
				input_weights.append(flat_vector[idx:idx+n_neurons].tolist())
				idx += n_neurons
			# Biases: one per neuron (neuron-major order)
			biases = flat_vector[idx:idx+n_neurons].tolist()
			idx += n_neurons
			# Build neuron-wise weights: [w1, w2, ...] for each neuron (no bias)
			layer_weights = []
			layer_biases = []
			for n in range(n_neurons):
				neuron_weights = [input_weights[i][n] for i in range(n_inputs)]
				layer_weights.append(neuron_weights)
				layer_biases.append(biases[n])
			weights_struct.append(layer_weights)
			biases_struct.append(layer_biases)
		return weights_struct, biases_struct

	def calculate_pso_feature_boundaries(self):
		layer_sizes = self.nn_config.layers_sizes
		activation_functions = self.nn_config.activation_functions
		input_dim = self.nn_config.input_dim
		full_layer_sizes = [input_dim] + layer_sizes
		boundaries = []
		for l in range(1, len(full_layer_sizes)):
			n_inputs = full_layer_sizes[l-1]
			n_neurons = full_layer_sizes[l]
			act_fn = activation_functions[l-1]
			weight_bounds = activation_boundary_weight[act_fn]
			bias_bounds = activation_boundary_bias[act_fn]
			for _ in range(n_inputs * n_neurons):
				boundaries.append(weight_bounds)
			for _ in range(n_neurons):
				boundaries.append(bias_bounds)
		return boundaries
	