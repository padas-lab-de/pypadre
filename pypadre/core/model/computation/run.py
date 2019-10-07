from typing import List

from pypadre.core.base import MetadataEntity, ChildEntity
from pypadre.core.model.generic.i_model_mixins import IStoreable, IProgressable
from pypadre.core.model.split.split import Split
from pypadre.core.printing.tablefyable import Tablefyable


class Run(IStoreable, IProgressable, MetadataEntity, ChildEntity, Tablefyable):

    SPLIT_IDS = "split_ids"

    @classmethod
    def _tablefy_register_columns(cls):
        pass

    def __init__(self, execution, parameter_selection: dict, splits=List[Split], **kwargs):
        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {}), **{self.SPLIT_IDS: [split.id for split in splits]}}
        super().__init__(schema_resource_name="run.json", parent=execution, result=self, metadata=metadata, **kwargs)
        self._parameter_selection = parameter_selection

    @property
    def execution(self):
        return self.parent

    @property
    def parameter_selection(self):
        return self._parameter_selection
