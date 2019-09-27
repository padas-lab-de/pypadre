from pypadre.core.base import MetadataEntity, ChildEntity
from pypadre.core.events.events import signals
from pypadre.core.model.generic.i_model_mixins import IStoreable, IProgressable
from pypadre.core.printing.tablefyable import Tablefyable


class Run(IStoreable, IProgressable, MetadataEntity, ChildEntity, Tablefyable):

    @classmethod
    def _tablefy_register_columns(cls):
        pass

    def __init__(self, execution, **kwargs):
        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {})}
        super().__init__(schema_resource_name="run.json", parent=execution, result=self, metadata=metadata, **kwargs)

    @property
    def execution(self):
        return self.parent
