from nn.nn import NeuralNetwork
from nn.entities import NNParams
from nn.constants import *
from data_prep.input_data_models import Point

nn = NeuralNetwork(NNParams(
    input_dim = 3,
    layers_sizes = [1],
    activation_functions = [ActFunc.LINEAR], 
    cost_function = CostFunc.MEAN_SQUARED_ERROR
))

weights = [[0.1, 0.2, 0.3]]
biases = [0.5]  
input_features = [
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 2.0]
]
targets = [1.0, 1.6, 0.0] 

# Create Point objects for the new interface
training_points = [
    Point(features_real_values=feat, target_real_value=targ,
          features_norm_values=feat, target_norm_value=targ)
    for feat, targ in zip(input_features, targets)
]

neuron_output = [sum(w * f for w, f in zip(weights[0], features)) + biases[0] for features in input_features]
actual_cost = sum((prediction - target) ** 2 for prediction, target in zip(neuron_output, targets)) / len(targets)

expected_cost, predictions = nn.forward_run_full_set(
    weights = [weights], 
    biases = [biases], 
    training_points = training_points
)   

print(f"Calculated cost : {expected_cost}")
print(f"Actual cost     : {actual_cost}")

assert abs(expected_cost - actual_cost) < 1e-6, f"Expected cost is {expected_cost} while actual cost is {actual_cost}"


