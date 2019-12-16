from pypadre.core.model.computation.computation import Computation


class Training(Computation):
    """ A execution should save data about the running env and the version of the code on which it was run """

    SPLIT_ID = "split_ID"

    @classmethod
    def _tablefy_register_columns(cls):
        super()._tablefy_register_columns()

    def __init__(self, split, model, **kwargs):
        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {self.SPLIT_ID: split.id})}
        super().__init__(schema_resource_name="model.json", result={"split": split, "model": model}, metadata=metadata, **kwargs)
        self._split = split
        self._model = model

    @property
    def model(self):
        return self._model

    @property
    def split(self):
        return self._split
