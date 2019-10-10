from pypadre.core.events.events import signals
from pypadre.core.model.computation.computation import Computation


class Evaluation(Computation):
    """ A execution should save data about the running env and the version of the code on which it was run """

    TRAINING_ID = "training_id"

    @classmethod
    def _tablefy_register_columns(cls):
        pass

    def __init__(self, *, training, result, **kwargs):
        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {}), **{self.TRAINING_ID: training.id}}
        self._training = training
        super().__init__(schema_resource_name="evaluation.json", predecessor=training, result=result, metadata=metadata, **kwargs)

    @property
    def estimation(self):
        return self._training
