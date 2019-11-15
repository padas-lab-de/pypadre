from pypadre._package import PACKAGE_ID
from pypadre.core.model.code.code_mixin import PythonPackage, Function
from pypadre.core.model.pipeline.parameter_providers.parameters import ParameterProvider


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


# class GridSearch(ProvidedCodeMixin, ParameterProviderMixin):
#     def __init__(self, **kwargs):
#         super().__init__(package=__name__, fn_name="_create_combinations", requirement=_name.__name__,
#                          version=_version.__version__, **kwargs)


# noinspection PyTypeChecker
grid_search = Function(fn=_create_combinations, transient=True,
                       identifier=PACKAGE_ID)


# Parameter provider holds a predefined code object but is itself something which has to be defined or managed somewhere
default_parameter_provider_ref = PythonPackage(package=__name__, variable="default_parameter_provider",
                                               identifier=PACKAGE_ID)
default_parameter_provider = ParameterProvider(name="default_provider", reference=default_parameter_provider_ref,
                                               code=grid_search)
