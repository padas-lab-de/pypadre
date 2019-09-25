from pypadre.core.base import MetadataEntity
from pypadre.core.events.events import signals
from pypadre.core.model.execution import Execution
from pypadre.core.model.generic.i_model_mixins import IProgressable, IStoreable
from pypadre.core.printing.tablefyable import Tablefyable


class Computation(IStoreable, IProgressable, MetadataEntity, Tablefyable):

    COMPONENT_ID = "component_id"
    EXECUTION_ID = "execution_id"

    @classmethod
    def _tablefy_register_columns(cls):
        pass

    def __init__(self, *, component, execution: Execution, result, **kwargs):
        super().__init__(metadata={**{self.COMPONENT_ID: component.id, self.EXECUTION_ID: execution.id}, **kwargs.pop("metadata", {})}, **kwargs)
        self._component = component
        self._execution = execution
        self._result = result

    # TODO Overwrite for no schema validation for now
    def validate(self, **kwargs):
        pass

    @property
    def component(self):
        return self._component

    @property
    def execution(self):
        return self._execution

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, result):
        self._result = result
