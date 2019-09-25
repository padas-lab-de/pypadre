from pypadre.core.base import MetadataEntity
from pypadre.core.events.events import signals
from pypadre.core.model.generic.i_model_mixins import IStoreable, IProgressable
from pypadre.core.printing.tablefyable import Tablefyable


class Run(IStoreable, IProgressable, MetadataEntity, Tablefyable):

    @classmethod
    def _tablefy_register_columns(cls):
        pass

    def __init__(self, execution, **kwargs):
        super().__init__(schema_resource_name="run.json", result=self, metadata=kwargs.pop("metadata", {}), **kwargs)
        self._execution = execution

    @property
    def execution(self):
        return self._execution
