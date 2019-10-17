from typing import List, Set

from pypadre.core.base import MetadataEntity, ChildEntity
from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.generic.i_model_mixins import IStoreable, IProgressable
from pypadre.core.model.split.split import Split
from pypadre.core.printing.tablefyable import Tablefyable


class PipelineOutput(IStoreable, MetadataEntity, ChildEntity, Tablefyable):

    SPLIT_IDS = "split_ids"

    @classmethod
    def _tablefy_register_columns(cls):
        pass

    def __init__(self, run, parameter_selection: dict, splits: Set[Split]=None, results=None, **kwargs):
        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {}), **{self.SPLIT_IDS: [split.id for split in splits]}}
        super().__init__(schema_resource_name="run.json", parent=run, result=self, metadata=metadata, **kwargs)
        self._parameter_selection = parameter_selection
        self._results = results

    @classmethod
    def from_computation(cls, computation: Computation):
        # Prepare parameter map for current computation
        parameter_selection = {computation.component.id: computation.parameters}

        # Prepare Set tracking all splits
        splits = set()

        # Get all parameters by looking at predecessors
        cur_computation = computation
        while cur_computation.predecessor is not None:
            cur_computation = cur_computation.predecessor
            parameter_selection[cur_computation.component.id] = cur_computation.parameters

            if isinstance(cur_computation, Split):
                splits.add(cur_computation)

        return cls(run=computation.run, parameter_selection=parameter_selection, splits=splits,
                   results=computation.result)

    @property
    def execution(self):
        return self.parent

    @property
    def results(self):
        return self._results

    @property
    def parameter_selection(self):
        return self._parameter_selection
