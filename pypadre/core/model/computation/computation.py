from pypadre.core.base import MetadataEntity, ChildEntity
from pypadre.core.model.computation.run import Run
from pypadre.core.model.generic.i_model_mixins import IProgressable, IStoreable
from pypadre.core.printing.tablefyable import Tablefyable


class Computation(IStoreable, IProgressable, MetadataEntity, ChildEntity, Tablefyable):
    COMPONENT_ID = "component_id"
    COMPONENT_CLASS = "component_class"
    RUN_ID = "run_id"

    @classmethod
    def _tablefy_register_columns(cls):
        pass

    def __init__(self, *, component, run: Run, result, **kwargs):
        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {}), **{self.COMPONENT_ID: component.id,
                                                                 self.COMPONENT_CLASS: str(component.__class__),
                                                                 self.RUN_ID: run.id}}

        super().__init__(parent=run, metadata=metadata, **kwargs)
        self._component = component
        self._result = result
        self.send_put()

    # TODO Overwrite for no schema validation for now
    def validate(self, **kwargs):
        pass

    @property
    def component(self):
        return self._component

    @property
    def run(self):
        return self.parent

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, result):
        self._result = result
