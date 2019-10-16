from typing import List

from pypadre.core.base import MetadataEntity, ChildEntity
from pypadre.core.model.generic.i_model_mixins import IStoreable, IProgressable
from pypadre.core.printing.tablefyable import Tablefyable
from pypadre.core.model.generic.i_executable_mixin import IExecuteable


class Run(IExecuteable, IStoreable, MetadataEntity, ChildEntity, Tablefyable):

    SPLIT_IDS = "split_ids"

    @classmethod
    def _tablefy_register_columns(cls):
        pass

    def __init__(self, execution,  **kwargs):
        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {})}
        super().__init__(schema_resource_name="run.json", parent=execution, result=self, metadata=metadata, **kwargs)

    def _execute_helper(self, *args, **kwargs):

        # Send signal
        self.send_put()

        # Start execution of the pipeline
        dataset = self.execution.experiment.dataset
        return self.pipeline.execute(dataset=dataset, run=self, **kwargs)

    @property
    def execution(self):
        return self.parent

    @property


    @property
    def dataset(self):
        return self.execution.dataset

    @property
    def experiment(self):
        return self.execution.experiment

    @property
    def pipeline(self):
        return self.execution.pipeline
