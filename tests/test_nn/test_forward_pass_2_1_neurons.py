from nn.nn import NeuralNetwork
from nn.entities import NNParams
from nn.constants import *
from data_prep.input_data_models import Point
import pytest

config = NNParams(
    input_dim = 3,
    layers_sizes = [2, 1],
    activation_functions = [ActFunc.RELU, ActFunc.LINEAR], 
    cost_function = CostFunc.MEAN_SQUARED_ERROR
)

# Test case data
input_features = [
    [1.0, 0.6, 1.0], 
    [1.0, 1.0, 1.0], 
    [0.0, 0.0, 2.0]
]
targets = [1.0, 1.2, 0.0]

# Create Point objects for the new interface
input_points = [
    Point(features_real_values=feat, target_real_value=targ,
          features_norm_values=feat, target_norm_value=targ)
    for feat, targ in zip(input_features, targets)
]

weights_layer0 = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.5]]
biases_layer0 = [0.2, 0.1]

weights_layer1 = [[0.5, 0.8]]
biases_layer1 = [0.5]

def test_nn_forward_pass() -> None:
    nn = NeuralNetwork(config)
    weights = [weights_layer0, weights_layer1]
    biases = [biases_layer0, biases_layer1]
    expected_nn_cost, predictions = nn.forward_run_full_set(weights, biases, input_points)
    actual_cost = cost()

    print(f"Expected NN Cost : {expected_nn_cost}")
    print(f"Actual Cost      : {actual_cost}")

    assert expected_nn_cost == pytest.approx(actual_cost, rel=1e-2)

def cost() -> float:
    nn_outputs = forward_pass_full_topology_all_points()
    cost = cost_calculation(nn_outputs)
    return cost

def cost_calculation(nn_outputs: list[float]) -> float:
    total_cost = 0.0
    if(config.cost_function == CostFunc.MEAN_SQUARED_ERROR):
        for output, target in zip(nn_outputs, targets):
            total_cost += (output - target) ** 2
        return total_cost / len(targets)
    else:
        raise ValueError("Unsupported cost function for this test.")

def forward_pass_full_topology_all_points() -> list[float]:
    return [forward_pass_topology(point) for point in input_features]

def forward_pass_topology(point) -> float:
    layer0_output = [
        forward_pass_neuron(point, weights_layer0[0], biases_layer0[0], ActFunc.RELU),
        forward_pass_neuron(point, weights_layer0[1], biases_layer0[1], ActFunc.RELU)
    ]
    layer1_output = forward_pass_neuron(layer0_output, weights_layer1[0], biases_layer1[0], ActFunc.LINEAR)
    return layer1_output

def forward_pass_neuron(p, w, b, act) -> float:
    s = sum(wi * pi for wi, pi in zip(w, p)) + b
    if act == ActFunc.RELU:
        return max(0, s)
    elif act == ActFunc.LINEAR:
        return s
    else:
        raise ValueError("Unsupported activation function for this test.")
    
if __name__ == "__main__":
    test_nn_forward_pass()
