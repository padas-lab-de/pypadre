"""
This file creates a JSON file for the mappings. It contains a detailed information about the different hyperparameters and the types of hyperparameters for the node classification experiment.
"""

import os
import json
from copy import deepcopy

algorithms = "algorithms"
model_parameters = "model_parameters"



name = "name"
other_names = "other_names"
implementation = "implementation"
description = "description"
hyper_parameters = "hyper_parameters"
node_classification = "node_classification"
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

node_classification_dict = dict()
node_classification_dict[name]= 'node_classification'
node_classification_dict[other_names] = []
node_classification_dict[implementation] = {'node_classification': 'padre.wrappers.wrapper_graphembeddings.WrapperNodeClassification'}
node_classification_dict[type_] = "Classifier"

#Classification Hyperparameters
classification_parameters_list = []

#strategy parameters
strategy_params_dict = dict()
strategy_params = 'strategy_params'
strategy_params_dict[name] =  strategy_params
strategy_params_dict[kind_of_value] = 'dictionary'
strategy_params_dict[optional] = True
strategy_params_dict[description] = 'The strategy chosen for the classifier'
strategy_params_implementation_dict = dict()
strategy_params_implementation_dict[path] = strategy_params

strategy_params_dict[node_classification] = strategy_params_implementation_dict
classification_parameters_list.append(deepcopy(strategy_params_dict))

#estimator_parameters
estimator_params_dict = dict()
estimator_params = 'estimator_params'
estimator_params_dict[name] = estimator_params
estimator_params_dict[kind_of_value] = 'dictionary'
estimator_params_dict[optional] = True
estimator_params_dict[description] = 'The estimator chosen for the classifier'
estimator_params_implementation_dict = dict()
estimator_params_implementation_dict[path] = estimator_params

estimator_params_dict[node_classification] = estimator_params_implementation_dict
classification_parameters_list.append(deepcopy(estimator_params_dict))

#metric parameters
metric_params_dict = dict()
metric_params = 'metric_params'
metric_params_dict[name] = metric_params
metric_params_dict[kind_of_value] = 'dictionary'
metric_params_dict[optional] = True
metric_params_dict[description] = 'The metric chosen for the evaluation'
metric_params_implementation_dict = dict()
metric_params_implementation_dict[path] = metric_params

metric_params_dict[node_classification] = metric_params_implementation_dict
classification_parameters_list.append(deepcopy(metric_params_dict))

hyperparameters_dict = dict()
hyperparameters_dict[model_parameters] = classification_parameters_list

node_classification_dict[hyper_parameters] = hyperparameters_dict

algorithms_list = []

algorithms_list.append(node_classification_dict)

algorithms_dict = dict()
algorithms_dict[algorithms]= algorithms_list

cwd = os.getcwd()

print(cwd)

with open('node_classification.json', 'w') as fp:
    json.dump(algorithms_dict,fp)



