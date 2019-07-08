class AttributeOnlyContainer:

    def __init__(self, attributes):
        self._attributes = attributes

    @property
    def attributes(self):
        return self._attributes

    @property
    def features(self):
        return None

    @property
    def targets(self):
        return None

    @property
    def data(self):
        return None

    @property
    def shape(self):
        return 0, 0

    @property
    def num_attributes(self):
        if self._attributes is None:
            return 0
        else:
            return len(self._attributes)


    def describe(self):
        return {"n_att" : len(self._attributes),
               "n_target" : len([a for a in self._attributes if a.is_target]),
                "stats": "no records available"}