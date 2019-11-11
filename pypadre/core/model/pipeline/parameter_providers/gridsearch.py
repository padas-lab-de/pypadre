from pypadre import _version, _name
from pypadre.core.model.generic.custom_code import ProvidedCodeMixin
from pypadre.core.model.pipeline.parameter_providers.parameters import ParameterProviderMixin


def _create_combinations(ctx, **parameters: dict):
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


class GridSearch(ProvidedCodeMixin, ParameterProviderMixin):
    def __init__(self, **kwargs):
        super().__init__(package=__name__, fn_name="_create_combinations",  requirement=_name.__name__, version=_version.__version__, **kwargs)
