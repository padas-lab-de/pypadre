from pypadre.core.model.computation.computation import Computation


class Training(Computation):
    """ A execution should save data about the running env and the version of the code on which it was run """

    @classmethod
    def _tablefy_register_columns(cls):
        pass

    def __init__(self, split, model, **kwargs):
        super().__init__(schema_resource_name="model.json", result=self, metadata=kwargs.pop("metadata", {}), **kwargs)
        self._split = split
        self._model = model

    @property
    def model(self):
        return self._model

    @property
    def split(self):
        return self._split
