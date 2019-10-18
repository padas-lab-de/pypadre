from abc import abstractmethod, ABCMeta

from pypadre.core.model.code.code import Code, EnvCode
from pypadre.core.model.computation.hyper_parameter_search import HyperParameterGrid


class ParameterMap:

    def __init__(self, parameter_dict):
        self._map = parameter_dict

    @property
    def map(self):
        return self._map

    def get_for(self, component):
        parameters = self.get(component.id)
        if len(parameters.keys()) == 0:
            if component.name in self._keys:
                return self.map.get(component.name)

            # Return empty dict
            return dict()

    def get(self, identifier):
        if identifier in self._keys:
            return self.map.get(identifier)
        return dict()

    def is_grid_search(self, component):
        pass

    @property
    def _keys(self):
        if self.map is None:
            return []

        return list(self.map.keys())


class IParameterProvider:
    __metaclass__ = ABCMeta

    @abstractmethod
    def combinations(self, *, run, component, predecessor, parameter_map: ParameterMap):
        raise NotImplementedError


class CodeParameterProvider(IParameterProvider):

    def __init__(self, code: Code):
        self._code = code

    @property
    def code(self):
        return self._code

    def combinations(self, *, run, component, predecessor, parameter_map: ParameterMap) -> HyperParameterGrid:
        # The function call will return a hyperparametergrid object
        return self._code.call(run=run, component=component,
                               predecessor=predecessor, parameter_map=parameter_map)


class GridSearch(EnvCode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, *args, run, component, predecessor, parameter_map: ParameterMap, **kwargs):
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
        parameters = parameter_map.get_for(component)

        # TODO parameters could also be a generator function if this is the case just call it and
        #  check if combinations are valid regarding the schema
        # TODO look through the parameters and add combination if one of it is a iterable
        #  instead of an expected parameter type
        # TODO expected parameter types are to be given in the component schema FIXME Christofer

        # If the parameters are returned within a function
        hyperparameters = parameters() if callable(parameters) else parameters
        assert (isinstance(hyperparameters, dict))

        # The params_list contains the names of the hyperparameters in the grid
        grid, params_list = self.create_combinations(hyperparameters)

        return HyperParameterGrid(component=component, run=run,
                                  parameters={},
                                  result=grid, parameter_names=params_list,
                                  predecessor=predecessor, branch=True)

    def create_combinations(self, parameters: dict):
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


class DefaultParameterProvider(CodeParameterProvider):

    def __init__(self):
        super().__init__(GridSearch())
