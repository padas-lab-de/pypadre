class WriteResultMetricsMap:

    def __init__(self, write_to_file: dict):
        self._map = write_to_file

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

