from pypadre.core.model.code.codemixin import Function
from pypadre.core.model.computation.hyper_parameter_search import HyperParameterGrid
from pypadre.core.model.generic.custom_code import CustomCodeHolder
from pypadre.core.util.inheritance import SuperStop


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

    @property
    def _keys(self):
        if self.map is None:
            return []

        return list(self.map.keys())


class ParameterProviderMixin(CustomCodeHolder, SuperStop):
    def _execute_helper(self, *args, run, component, predecessor=None, parameter_map, **kwargs):
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
        grid, params_list = super()._execute_helper(run=run, component=component, predecessor=predecessor,
                                              parameter_map=parameter_map, parameters=hyperparameters)

        return HyperParameterGrid(component=component, run=run,
                                  parameters={},
                                  result=grid, parameter_names=params_list,
                                  predecessor=predecessor, branch=True)


class FunctionParameterProvider(ParameterProviderMixin):

    def __init__(self, *args, fn, **kwargs):
        super().__init__(*args, code=Function(fn=fn), **kwargs)
