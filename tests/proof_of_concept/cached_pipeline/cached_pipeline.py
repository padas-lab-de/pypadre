from pypadre.pod.experimentcreator import ExperimentCreator
from copy import deepcopy
import itertools
import numpy as np


def normal_fit(workflow, features, targets, parameters=None):
    """
    This function fits the workflow with all the parameters in an iterative manner.

    :param workflow: The pipeline to be executed
    :param features: features to be fit
    :param targets: labels of the corresponding features
    :param parameters: parameters to be used to fit

    :return:
    """
    master_list = []
    params_list = []
    for estimator in parameters:
        param_dict = parameters.get(estimator)
        for params in param_dict:
            master_list.append(param_dict.get(params))
            params_list.append(''.join([estimator, '.', params]))
    grid = itertools.product(*master_list)

    # For each tuple in the combination create a run
    for element in grid:
        # Get all the parameters to be used on set_param
        for param, idx in zip(params_list, range(0, len(params_list))):
            split_params = param.split(sep='.')
            estimator = workflow.named_steps.get(split_params[0])

            if estimator is None:
                print(f"Estimator {split_params[0]} is not present in the pipeline")
                break

            estimator.set_params(**{split_params[1]: element[idx]})

        workflow.fit(features, targets)


def optimized_workflow(workflow, features, targets, parameters=None):
    """
    This function implements an optimized workflow function

    :param workflow: The pipeline to be executed
    :param features: features to be fit
    :param target: labels to the corresponding features
    :param parameters: parameters to be used to fit

    :return:
    """
    steps = []

    # Compute the parameter combinations as dictionaries for all the estimators in their order in the pipeline
    # Store these results in a dictionary with the estimator name as the key
    # Create a array of parameter combinations where each element in the array is a stack
    # Call fit or fit transform for a single parameter combination to populate the result list
    # Go the last estimator
    # Repeat until all the stacks are empty

    # Get the names of the estimators in the order they are executed
    for step in workflow.steps:
        steps.append(step[0])

    params_stack = [None] * len(steps)
    results = [None] * len(steps)
    results[0] = features

    estimator_index = 0
    for step in steps:
        param_stack = []
        estimator_params = parameters.get(step, None)
        if estimator_params is None:
            params_stack[estimator_index] = param_stack
            estimator_index += 1
            continue

        if len(estimator_params.keys()) == 1:
            # Populate a dictionary array containing the parameter name, parameter value combination
            # This is used to set the value to the estimator
            param_name = list(estimator_params.keys())[0]
            params = estimator_params.get(param_name)
            for param in params:
                curr_param = dict()
                curr_param[param_name] = param
                param_stack.append(curr_param)

            params_stack[estimator_index] = param_stack

        else:
            # Multiple parameters exist for the same estimator.
            # Need to find all possible combinations of the parameters
            master_list = []
            params_list = []
            for params in estimator_params:
                # Append the list of parameters to the master list
                master_list.append(estimator_params.get(params))
                # Append the parameter name to the list
                params_list.append(params)

            # Compute the combinations of all the parameters
            grid = itertools.product(*master_list)

            # For every element on the grid create a dictionary and add it to the parameter stack
            param_stack = []
            limit = len(params_list)
            for element in grid:
                # Create a combination of parameters
                param_combination = dict()
                for idx in range(limit):
                    param_combination[params_list[idx]] = element[idx]

                # Add the combination to the stack
                param_stack.append(param_combination)

            params_stack[estimator_index] = param_stack

        estimator_index += 1

    # Create an initial stack array for executing
    execution_stack = deepcopy(params_stack)
    continue_execution = True
    while continue_execution:

        if results[-1] is None:
            # Backtrack until a valid results matrix is found.
            # From the next estimator onwards pop a parameter combination, fit_transform and then push the results
            # Copy all results from that estimator till the end
            idx = [idx for idx in range(len(results)) if results[idx] is not None][-1]
            # Fit transform function to be applied until last estimator
            while idx < len(execution_stack) - 1:
                params = execution_stack[idx].pop()
                workflow.steps[idx][-1].set_params(**params)
                results[idx + 1] = workflow.steps[idx][-1].fit_transform(features)
                idx += 1

        else:
            # pop a single parameter combination if it is possible and fit it
            # if there are no parameters to be set for the parameter backtrack until a non empty param stack is found
            # Pop a parameter combination and set it to the estimator
            # Copy all parameters from that point to end
            # Execute fit function from that point.
            # Copy results of the fit to estimator_idx + 1 position
            if len(execution_stack[-1]) > 0:
                # Pop parameters
                params = execution_stack[-1].pop()
                # Set the parameters to the estimator
                workflow.steps[-1][-1].set_params(**params)

            else:
                # Search for the first non empty index in the execution parameters list
                # If all parameters are exhausted break
                # Return the list of non empty indices
                non_empty_indices = [idx for idx in range(len(execution_stack)) if execution_stack[idx] and
                                     isinstance(execution_stack[idx], list)]

                # if all indices are empty stop execution
                if not non_empty_indices:
                    continue_execution = False
                    break

                idx = non_empty_indices[-1]

                # Fit transform function to be applied until last estimator
                while idx < len(execution_stack) - 1:
                    params = execution_stack[idx].pop()
                    workflow.steps[idx][-1].set_params(**params)
                    results[idx + 1] = workflow.steps[idx][-1].fit_transform(features)
                    # Since all the other estimators from that point onwards will be reset,
                    # copy the parameters of estimators to the execution stack
                    execution_stack[idx + 1] = deepcopy(params_stack[idx + 1])
                    idx += 1

        # Fit the last estimator
        workflow.steps[-1][-1].fit(features, targets)


def main():
    experiment_creator = ExperimentCreator()
    # FIRST TEST EXPERIMENT WITH MULTIPLE DATASETS
    params = {'n_components': [2, 3, 4, 5, 6, 7]}
    params_svr = {'C': [0.5, 0.7, 0.8, 0.9, 1.0, 1.5],
                  'degree': [1, 2, 3, 4, 5, 6]}
    param_value_dict = {'principal component analysis': params, 'epsilon-support vector regression':params_svr}
    workflow = experiment_creator.create_test_pipeline(['PCA', 'SVR'])
    dataset = experiment_creator.get_local_dataset('Diabetes')

    import time
    c1 = time.time()
    normal_fit(workflow=workflow, features=dataset.features(),
               targets=np.reshape(dataset.targets(), [dataset.targets().shape[0]]),
               parameters=param_value_dict)
    c2 = time.time()
    print('Execution time for normal workflow:{time_diff}'.format(time_diff=c2 - c1))

    c1 = time.time()
    optimized_workflow(workflow=workflow, features=dataset.features(),
                       targets=np.reshape(dataset.targets(), [dataset.targets().shape[0]]),
                       parameters=param_value_dict)
    c2 = time.time()
    print('Execution time for optimized workflow:{time_diff}'.format(time_diff=c2 - c1))

    return


if __name__ == '__main__':
    main()
