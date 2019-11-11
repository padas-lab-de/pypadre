from pypadre import _version, _name
from pypadre.core.model.generic.custom_code import ProvidedCodeMixin
from pypadre.core.model.pipeline.parameter_providers.parameters import ParameterProviderMixin


def _create_combinations(ctx, **parameters: dict):
    import itertools

    # Generate every possible combination of the provided hyper parameters.
    master_list = []
    params_list = []
    for estimator in parameters:
        param_dict = parameters.get(estimator)
        # assert_condition(condition=isinstance(param_dict, dict),
        #                  source=self,
        #                  source=self,
        #                  message='Parameter dictionary is not of type dictionary for estimator:' + estimator)
        for params in param_dict:
            # Append only the parameters to create a master list
            master_list.append(param_dict.get(params))

            # Append the estimator name followed by the parameter to create a ordered list.
            # Ordering of estimator.parameter corresponds to the value in the resultant grid tuple
            params_list.append(''.join([estimator, '.', params]))

    #TODO christofer what is with single values? ex: return {'SKLearnEstimator': {'parameters': {'SVC': {'C': 0.5}}}}
    grid = itertools.product(*master_list)
    return grid, params_list


# def grid_search(ctx, **kwargs):
#     """
#        # We need to either create multiple components
#        # based on the number of elements in the grid or iterate of the grid
#        for element in grid:
#
#            # Set the params to the component either via a dictionary all together or individually
#            execution_params = dict()
#            for param, idx in zip(params_list, range(0, len(params_list))):
#                execution_params[param] = element[idx]
#
#            # TODO set the parameters to the component
#            # yield Computation(component=component, execution=execution, parameters=execution_params, branch=False)
#            # TODO Decide whether the grid creation logic should be within the HyperParameter Search Component or not
#            yield HyperParameterSearch(component=component, execution=execution,
#                                       parameters=execution_params, predecessor=predecessor, branch=False)
#        """
#     (run, component, predecessor, parameter_map) = \
#         unpack(ctx, "run", "component", ("predecessor", None), "parameter_map")
#     parameter_map: ParameterMap
#
#     parameters = parameter_map.get_for(component)
#
#     #  check if combinations are valid regarding the schema
#     # TODO look through the parameters and add combination if one of it is a iterable
#     #  instead of an expected parameter type
#     # TODO expected parameter types are to be given in the component schema FIXME Christofer
#
#     # If the parameters are returned within a function
#     hyperparameters = parameters() if callable(parameters) else parameters
#     assert (isinstance(hyperparameters, dict))
#
#     # The params_list contains the names of the hyperparameters in the grid
#     grid, params_list = _create_combinations(hyperparameters)
#
#     return HyperParameterGrid(component=component, run=run,
#                               parameters={},
#                               result=grid, parameter_names=params_list,
#                               predecessor=predecessor, branch=True)


class SKLearnGridSearch(ProvidedCodeMixin, ParameterProviderMixin):
    def __init__(self, **kwargs):
        super().__init__(package=__name__, fn_name="_create_combinations",  requirement=_name.__name__, version=_version.__version__, **kwargs)
