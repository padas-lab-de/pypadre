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
boolean = 'boolean'
str_ = 'string'
range = "range"
range_greater_than_one = "Z >= 1"
range_bool = '{True, False}'
optional = "optional"


pytorch_dict  = dict()
pytorch_dict[name] = 'pytorch'
pytorch_dict[other_names] = []
pytorch_dict[wikidata_id] = "Q47509047"
pytorch_dict[implementation] = {'pytorch':'padre.wrappers.wrapper_pytorch.WrapperPytorch'}
pytorch_dict[type_] = "Neural Network"

ratio = "ratio"
nominal = "nominal"
interval = "interval"
ordinal = "ordinal"

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
steps_dict[measurement_scale] = ratio

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
batch_size_dict[description] = "Number of samples acquired for each iteration"
batch_size_dict[measurement_scale] = ratio

batch_size_implementation_dict = dict()
batch_size_implementation_dict[path] = batch_size
batch_size_implementation_dict[default_value] = 1

batch_size_dict[pytorch] = batch_size_implementation_dict
model_parameters_list.append(deepcopy(batch_size_dict))

# Checkpoint
checkpoint_dict = dict()
checkpoint = "checkpoint"
checkpoint_dict[name] = checkpoint
checkpoint_dict[kind_of_value] = integer_
checkpoint_dict[range] = range_greater_than_one
checkpoint_dict[optional] = True
checkpoint_dict[description] = "A save point for the intermediate trained model is created every N number of iterations"
checkpoint_dict[measurement_scale] = ratio

checkpoint_implementation_dict = dict()
checkpoint_implementation_dict[path] = checkpoint
checkpoint_implementation_dict[default_value] = None

checkpoint_dict[pytorch] = checkpoint_implementation_dict
model_parameters_list.append(deepcopy(checkpoint_dict))

# Resume
resume_dict = dict()
resume = 'resume'
resume_dict[name] = resume
resume_dict[kind_of_value] = boolean
resume_dict[range] = range_bool
resume_dict[optional] = True
resume_dict[description] = "Allows the model to resume training based on an earlier saved check point"
resume_dict[measurement_scale] = nominal

resume_implementation_dict = dict()
resume_implementation_dict[path] = resume
resume_implementation_dict[default_value] = False

resume_dict[pytorch] = resume_implementation_dict
model_parameters_list.append(deepcopy(resume_dict))

# Pre trained model path
pre_trained_model_path_dict = dict()
pre_trained_model_path = 'pre_trained_model_path'
pre_trained_model_path_dict[name] = pre_trained_model_path
pre_trained_model_path_dict[kind_of_value] = str_
# Range is not specified
pre_trained_model_path_dict[optional] = True
pre_trained_model_path_dict[description] = "Path of a saved model to be loaded for continuing the training"
pre_trained_model_path_dict[measurement_scale] = ordinal

pre_trained_model_path_implementation_dict = dict()
pre_trained_model_path_implementation_dict[path] = pre_trained_model_path
pre_trained_model_path_implementation_dict[default_value] = None

pre_trained_model_path_dict[pytorch] = pre_trained_model_path_implementation_dict
model_parameters_list.append(deepcopy(pre_trained_model_path_dict))

# Model prefix
model_prefix_dict = dict()
model_prefix = 'model_prefix'
model_prefix_dict[name] = model_prefix
model_prefix_dict[kind_of_value] = str_
# Range is not specified as it is a string
model_prefix_dict[optional] = True
model_prefix_dict[description] = "The prefix string to be given to the name of the model"
model_prefix_dict[measurement_scale] = ordinal

model_prefix_implementation_dict = dict()
model_prefix_implementation_dict[path] = model_prefix
model_prefix_implementation_dict[default_value] = model_prefix

model_prefix_dict[pytorch] = model_prefix_implementation_dict
model_parameters_list.append(deepcopy(model_prefix_dict))

'''
loss function
lr scheduler
self.architecture = params.get('architecture', None)
'''

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
