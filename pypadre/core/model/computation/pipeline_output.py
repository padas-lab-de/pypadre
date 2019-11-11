from typing import Set

from pypadre.core.base import MetadataMixin, ChildMixin
from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.generic.i_model_mixins import StoreableMixin
from pypadre.core.model.split.split import Split
from pypadre.core.printing.tablefyable import Tablefyable


class PipelineOutput(StoreableMixin, MetadataMixin, ChildMixin, Tablefyable):

    SPLIT_IDS = "split_ids"

    @classmethod
    def _tablefy_register_columns(cls):
        pass

    def __init__(self, run, parameter_selection: dict, metrics: dict, splits: Set[Split]=None, results=None, **kwargs):
        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {}), **{self.SPLIT_IDS: [split.id for split in splits]}}
        super().__init__(schema_resource_name="run.json", parent=run, result=self, metadata=metadata, **kwargs)
        self._parameter_selection = parameter_selection
        self._metrics = metrics
        self._results = results

    @classmethod
    def from_computation(cls, computation: Computation):
        # Prepare parameter map for current computation
        parameter_selection = {computation.component.id: computation.parameters}
        metrics = {computation.component.id: [m.result for m in computation.metrics]}

        # Prepare Set tracking all splits
        splits = set()

        # Get all parameters by looking at predecessors
        cur_computation = computation
        while cur_computation.predecessor is not None:
            cur_computation = cur_computation.predecessor
            parameter_selection[cur_computation.component.id] = cur_computation.parameters
            metrics[cur_computation.component.id] = cur_computation.metrics

            if isinstance(cur_computation, Split):
                splits.add(cur_computation)

        return cls(run=computation.run, parameter_selection=parameter_selection, metrics=metrics, splits=splits,
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

    @property
    def metrics(self):
        return self._metrics
