from pypadre.core.events.events import signals
from pypadre.core.model.computation.computation import Computation


class Evaluation(Computation):
    """ A execution should save data about the running env and the version of the code on which it was run """

    @classmethod
    def _tablefy_register_columns(cls):
        pass

    def __init__(self, *, training, **kwargs):
        super().__init__(schema_resource_name="evaluation.json", result=self, metadata=kwargs.pop("metadata", {}), **kwargs)
        self._training = training

    @property
    def estimation(self):
        return self._training
