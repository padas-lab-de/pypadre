from abc import abstractmethod
from typing import Type

from pypadre.core.model.code.code_mixin import CodeMixin
from pypadre.core.model.computation.hyper_parameter_search import HyperParameterGrid
from pypadre.core.model.generic.custom_code import CustomCodeHolder, CodeManagedMixin
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


class ParameterProviderMixin(CodeManagedMixin, CustomCodeHolder, SuperStop):

    @abstractmethod
    def __init__(self, *args, reference: Type[CodeMixin] = None, **kwargs):
        super().__init__(*args, reference=reference, **kwargs)

    def _execute_helper(self, *args, run, component, predecessor=None, parameter_map, **kwargs):

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


class ParameterProvider(ParameterProviderMixin):
    """
    Generic parameter provider.
    """

    def __init__(self, name="custom_provider", code=None, **kwargs):
        # TODO wrap in something better than function and get variables
        super().__init__(name=name, code=code, **kwargs)
