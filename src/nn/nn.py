from typing import List
import numpy as np
import math
from typing import List
from nn.constants import *
from nn.entities import NNParams
from data_prep.input_data_models import Point

class Neuron:
    def __init__(self, activation_function: ActFunc):
        self.activation_function = activation_function
        self.output = None
        self.weighted_sum = None
    
    def _forward(self, inputs: List[float], weights: List[float], bias: float) -> float:
        self.weighted_sum = bias + sum(i * w for i, w in zip(inputs, weights))
        self.output = self._apply_activation(self.weighted_sum)
        return self.output
    
    def _apply_activation(self, x: float) -> float:
        if self.activation_function == ActFunc.SIGMOID:
            return 1.0 / (1.0 + math.exp(-np.clip(x, -500, 500)))
        elif self.activation_function == ActFunc.RELU:
            return max(0.0, x)
        elif self.activation_function == ActFunc.TANH:
            return math.tanh(x)
        elif self.activation_function == ActFunc.LINEAR:
            return x
        else:
            raise ValueError(f"Unknown activation function: {self.activation_function}")
 
class Layer:
    def __init__(self, neurons_count: int, activation_function: ActFunc):
        self.neurons = [Neuron(activation_function) for _ in range(neurons_count)]
        self.outputs = []
    
    def _forward(self, inputs: List[float], weights: List[List[float]], biases: List[float]) -> List[float]:
        self.outputs = []
        for neuron_idx, neuron in enumerate(self.neurons):
            neuron_output = neuron._forward(inputs, weights[neuron_idx], biases[neuron_idx])
            self.outputs.append(neuron_output)
        return self.outputs

class NeuralNetwork:
    def __init__(self, config: NNParams):
        self.layers_sizes = config.layers_sizes
        self.cost_function = config.cost_function
        self.layers = []

        # Create layers (input layer in not included in the config.layers_sizes)
        for i in range(0, len(config.layers_sizes)):
            layer = Layer(config.layers_sizes[i], config.activation_functions[i])
            self.layers.append(layer)

    def _set_weights_and_biases(self, weights: List[List[List[float]]], biases: List[List[float]]) -> None:
        self.weights = weights
        self.biases = biases

    def _forward_run_one_point(self, point: List[float]) -> float:
        current_input = point
        for layer_idx, layer in enumerate(self.layers):
            layer_weights = self.weights[layer_idx]
            layer_biases = self.biases[layer_idx]
            current_input = layer._forward(current_input, layer_weights, layer_biases)
        return current_input[0]  # Assuming single output neuron
    
    def forward_run_full_set(self, 
                    weights: List[List[List[float]]], 
                    biases: List[List[float]],
                    training_points: List[Point]) -> tuple[float, np.ndarray]:
        # Merged implementation - get predictions and calculate cost in one method
        self._set_weights_and_biases(weights, biases)
        predictions = []
        for point in training_points:
            predictions.append(self._forward_run_one_point(point.features_norm_values))
        
        predictions_array = np.array(predictions)
        targets = [point.target_norm_value for point in training_points]
        total_cost = self._apply_cost_function(predictions, targets)
        return total_cost, predictions_array
    
    def _apply_cost_function(self, predictions: List[float], targets: List[float]) -> float:
        if self.cost_function == CostFunc.MEAN_SQUARED_ERROR:
            return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(targets)
        elif self.cost_function == CostFunc.MEAN_ABSOLUTE_ERROR:
            return sum(abs(p - t) for p, t in zip(predictions, targets)) / len(targets)
        else:
            raise ValueError(f"Unknown cost function: {self.cost_function}")
        
        


