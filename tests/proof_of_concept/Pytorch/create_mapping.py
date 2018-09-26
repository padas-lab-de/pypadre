"""
This file creates a JSON file for the mappings. It contains a detailed information about the different hyperparameters and the types of hyperparameters
"""
import os
import json

algorithms = "algorithms"
model_parameters = "model_parameters"
optimization_parameters = "optimization_parameters"
execution_parameters = "execution_parameters" \
                       ""
name = "name"
other_names = "other_names"
wikidata_id = "wikidata_id"
implementation = "implementation"
description = "description"
measurement_scale = "measurement_scale"
hyper_parameters = "hyper_parameters"
pytorch = "pytorch"
path = "path"
default_value = "default_value"
type_ = "type"
kind_of_value = "kind_of_value"
integer_ = "integer"
range = "range"
range_greater_than_one = "Z >= 1"
optional = "optional"


pytorch_dict  = dict()
pytorch_dict[name] = 'pytorch'
pytorch_dict[other_names] = []
pytorch_dict[wikidata_id] = "Q47509047"
pytorch_dict[implementation] = {'pytorch':'padre.pypadre.wrappers.wrapper_pytorch'}
pytorch_dict[type_] = "Neural Network"

# Model Hyperparameters
model_parameters_list = []

steps_dict = dict()
steps = "steps"
steps_dict[name] = steps
steps_dict[kind_of_value] = integer_
steps_dict[range] = range_greater_than_one
steps_dict[optional] = False
steps_dict[description] = "Number of iterations that the data should pass through the Neural Network"
steps_dict[measurement_scale] = "interval"

steps_implementation_dict = dict()
steps_implementation_dict[path] = steps
steps_implementation_dict[default_value] = 100

steps_dict[pytorch] = steps_implementation_dict
model_parameters_list.append(steps_dict)

# Model optimization parameters
optimization_parameters_list = []

# Model execution parameters
execution_parameters_list = []

hyperparameters_dict = dict()
hyperparameters_dict[model_parameters] = model_parameters_list
hyperparameters_dict[optimization_parameters] = optimization_parameters_list
hyperparameters_dict[execution_parameters] = execution_parameters_list

pytorch_dict[hyper_parameters] = hyperparameters_dict

algorithms_list = []
algorithms_list.append(pytorch_dict)

algorithms_dict = dict()
algorithms_dict[algorithms] = algorithms_list

cwd = os.getcwd()
print(cwd)
with open('torch.json', 'w') as fp:
    json.dump(algorithms_dict, fp)
