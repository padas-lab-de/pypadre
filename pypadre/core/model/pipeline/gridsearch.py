from pypadre import _version, _name
from pypadre.core.model.computation.hyper_parameter_search import HyperParameterGrid
from pypadre.core.model.generic.custom_code import IProvidedCode
from pypadre.core.model.pipeline.parameters import IParameterProvider, ParameterMap
from pypadre.core.util.utils import unpack


def _create_combinations(parameters: dict):
    """
    Creates all the possible combinations of hyper parameters passed
    :param parameters: Dictionary containing hyperparameter names and their possible values
    :return: A list containing all the combinations and a list containing the hyperparameter names
    """

    import itertools

    params_list = []
    master_list = []

    for parameter in parameters:
        # Append only the parameters to create a master list
        parameter_values = parameters.get(parameter)

        # If the parameter value is a dict wrap it in a dictionary,
        # so that the values of the dictionary are not unpacked
        parameter_values = [parameter_values] if isinstance(parameter_values, dict) else parameter_values

        master_list.append(parameter_values)

        # Append the estimator name followed by the parameter to create a ordered list.
        # Ordering of estimator.parameter corresponds to the value in the resultant grid tuple
        params_list.append(parameter)

    # Create the grid
    grid = itertools.product(*master_list)

    return grid, params_list


def grid_search(ctx, **kwargs):
    """
       # We need to either create multiple components
       # based on the number of elements in the grid or iterate of the grid
       for element in grid:

           # Set the params to the component either via a dictionary all together or individually
           execution_params = dict()
           for param, idx in zip(params_list, range(0, len(params_list))):
               execution_params[param] = element[idx]

           # TODO set the parameters to the component
           # yield Computation(component=component, execution=execution, parameters=execution_params, branch=False)
           # TODO Decide whether the grid creation logic should be within the HyperParameter Search Component or not
           yield HyperParameterSearch(component=component, execution=execution,
                                      parameters=execution_params, predecessor=predecessor, branch=False)
       """
    (run, component, predecessor, parameter_map) = \
        unpack(ctx, "run", "component", ("predecssor", None), "parameter_map")
    parameter_map: ParameterMap

    parameters = parameter_map.get_for(component)

    #  check if combinations are valid regarding the schema
    # TODO look through the parameters and add combination if one of it is a iterable
    #  instead of an expected parameter type
    # TODO expected parameter types are to be given in the component schema FIXME Christofer

    # If the parameters are returned within a function
    hyperparameters = parameters() if callable(parameters) else parameters
    assert (isinstance(hyperparameters, dict))

    # The params_list contains the names of the hyperparameters in the grid
    grid, params_list = _create_combinations(hyperparameters)

    return HyperParameterGrid(component=component, run=run,
                              parameters={},
                              result=grid, parameter_names=params_list,
                              predecessor=predecessor, branch=True)


class GridSearch(IProvidedCode, IParameterProvider):
    def __init__(self, **kwargs):
        super().__init__(package=__name__, fn_name="grid_search",  requirement=_name.__name__, version=_version.__version__, **kwargs)
