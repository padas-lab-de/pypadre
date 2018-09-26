"""
This file creates a JSON file for the mappings. It contains a detailed information about the different hyperparameters and the types of hyperparameters
"""
import os
import json
from copy import deepcopy

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
pytorch_dict[implementation] = {'pytorch':'padre.wrappers.wrapper_pytorch.WrapperPytorch'}
pytorch_dict[type_] = "Neural Network"

# Model Hyperparameters
model_parameters_list = []

# Steps
steps_dict = dict()
steps = "steps"
steps_dict[name] = steps
steps_dict[kind_of_value] = integer_
steps_dict[range] = range_greater_than_one
steps_dict[optional] = False
steps_dict[description] = "Number of iterations that the Neural network should be optimized by backward pass"
steps_dict[measurement_scale] = "ratio"

steps_implementation_dict = dict()
steps_implementation_dict[path] = steps
steps_implementation_dict[default_value] = 100

steps_dict[pytorch] = steps_implementation_dict
model_parameters_list.append(deepcopy(steps_dict))

# Batch Size
batch_size_dict = dict()
batch_size = "batch_size"
batch_size_dict[name] = batch_size
batch_size_dict[kind_of_value] = integer_
batch_size_dict[range] = range_greater_than_one
batch_size_dict[optional] = False
batch_size_dict[description] = ""
batch_size_dict[measurement_scale] = "ratio"

batch_size_implementation_dict = dict()
batch_size_implementation_dict[path] = batch_size
batch_size_implementation_dict[default_value] = 1

batch_size_dict[pytorch] = batch_size_implementation_dict
model_parameters_list.append(deepcopy(batch_size_dict))


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
