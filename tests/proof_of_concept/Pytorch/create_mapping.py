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
type_ = "type"
kind_of_value = "kind_of_value"
integer = "integer"
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
steps = "steps"
model_parameters_dict = dict()
model_parameters_dict[name] = steps
model_parameters_dict[kind_of_value] = integer
model_parameters_dict[range] = range_greater_than_one
model_parameters_dict[optional] = False

# Model optimization parameters
optimization_parameters_dict = dict()

# Model execution parameters
execution_parameters_dict = dict()

hyperparameters_dict = dict()
hyperparameters_dict[model_parameters] = model_parameters_dict
hyperparameters_dict[optimization_parameters] = optimization_parameters_dict
hyperparameters_dict[execution_parameters] = execution_parameters_dict

algorithms_dict = dict()
algorithms_dict[algorithms] = pytorch_dict

cwd = os.getcwd()
print(cwd)
with open('torch.json', 'w') as fp:
    json.dump(algorithms_dict, fp)
