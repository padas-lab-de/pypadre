from abc import ABCMeta

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

    def is_grid_search(self, component):
        pass

    @property
    def _keys(self):
        if self.map is None:
            return []

        return list(self.map.keys())


class IParameterProvider(CustomCodeHolder, SuperStop):
    __metaclass__ = ABCMeta
