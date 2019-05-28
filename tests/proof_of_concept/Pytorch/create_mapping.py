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
metadata = 'metadata'
author = 'author'
library = 'library'
library_version = 'library_version'
mapping_version = 'mapping_version'

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
list_ = 'list'
range = "range"
range_greater_than_one = "Z >= 1"
range_bool = '{True, False}'
optional = "optional"


pytorch_dict = dict()
pytorch_dict[name] = 'pytorch'
pytorch_dict[other_names] = []
pytorch_dict[wikidata_id] = "Q47509047"
pytorch_dict[implementation] = {'pytorch': 'padre.core.wrappers.wrapper_pytorch.WrapperPytorch'}
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

# Optimizer
optimizer_params_dict = dict()
optimizer_params = 'optimizer_params'
optimizer_params_dict[name] = optimizer_params
optimizer_params_dict[kind_of_value] = "dictionary"

# Range is not specified as it is a dictionary
optimizer_params_dict[optional] = False
optimizer_params_dict[description] = 'The optimizer method chosen for the model'

optimizer_params_implementation_dict = dict()
optimizer_params_implementation_dict[path] = optimizer_params

optimizer_params_dict[pytorch] = optimizer_params_implementation_dict
model_parameters_list.append(deepcopy(optimizer_params_dict))

# lr scheduler

lr_scheduler_params_dict = dict()
lr_scheduler_params = 'lr_scheduler_params'
lr_scheduler_params_dict[name] = lr_scheduler_params
lr_scheduler_params_dict[kind_of_value] = 'dictionary containing the state of the lr_scheduler'
# Range is not specified as it is a dict
lr_scheduler_params_dict[optional] = True
lr_scheduler_params_dict[description] = 'The dictionary contains the different parameters of the lr scheduler'

lr_scheduler_params_implementation_dict = dict()
lr_scheduler_params_implementation_dict[path] = lr_scheduler_params

lr_scheduler_params_dict[pytorch] = lr_scheduler_params_implementation_dict
model_parameters_list.append(deepcopy(lr_scheduler_params_dict))

# loss_function
loss_params_dict = dict()
loss_params = 'loss_params'
loss_params_dict[name] = loss_params
loss_params_dict[kind_of_value] = 'dictionary containing the state of the loss function'
# Range is not specified
loss_params_dict[optional] = False
loss_params_dict[description] = 'The dictionary contains the different parameters of the loss function'

loss_params_implementation_dict = dict()
loss_params_implementation_dict[path] = loss_params

loss_params_dict[pytorch] = loss_params_implementation_dict
model_parameters_list.append(deepcopy(loss_params_dict))

# layer order
layer_order_dict = dict()
layer_order = 'layer_order'
layer_order_dict[name] = layer_order
layer_order_dict[kind_of_value] = list_
# Range is not specified
layer_order_dict[optional] = False
layer_order_dict[description] = 'This describes the ordering of the different layers of the network'

layer_order_implementation_dict = dict()
layer_order_implementation_dict[path] = layer_order

layer_order_dict[pytorch] = layer_order_implementation_dict
model_parameters_list.append(deepcopy(layer_order_dict))

# Architecture
architecture_dict = dict()
architecture = 'architecture'
architecture_dict[name] = architecture
architecture_dict[kind_of_value] = 'dictionary containing the layer names and its parameters'
# Range is not specified
architecture_dict[optional] = False
architecture_dict[description] = 'The dictionary contains the semantic description of each of the layers'

architecture_implementation_dict = dict()
architecture_implementation_dict[path] = architecture

architecture_dict[pytorch] = architecture_implementation_dict
model_parameters_list.append(deepcopy(architecture_dict))

'''
layers
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

metadata_dict = dict()
metadata_dict[author] = 'Michael Granitzer'
metadata_dict[library] = 'pytorch'
metadata_dict[library_version] = '0.4.0'
metadata_dict[mapping_version] = '0.1'


algorithms_list = list()
algorithms_list.append(pytorch_dict)

algorithms_dict = dict()
algorithms_dict[algorithms] = algorithms_list
algorithms_dict[metadata] = metadata_dict

cwd = os.getcwd()
print(cwd)
with open('pytorch.json', 'w') as fp:
    json.dump(algorithms_dict, fp)
