from pypadre.core.base import MetadataEntity, ChildEntity
from pypadre.core.events.events import signals
from pypadre.core.model.generic.i_model_mixins import IStoreable, IProgressable, IExecuteable
from pypadre.core.model.pipeline.parameters import PipelineParameters
from pypadre.core.printing.tablefyable import Tablefyable


class Execution(IStoreable, IProgressable, IExecuteable, MetadataEntity, ChildEntity, Tablefyable):
    """ A execution should save data about the running env and the version of the code on which it was run """

    EXPERIMENT_ID = "experiment_id"

    @classmethod
    def _tablefy_register_columns(cls):
        # Add entries for tablefyable
        cls.tablefy_register_columns({'hash': 'hash', 'cmd': 'cmd'})

    def __init__(self, experiment, codehash=None, command=None, **kwargs):
        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults, **{self.EXPERIMENT_ID: experiment.id}, **kwargs.pop("metadata", {})}

        super().__init__(parent=experiment, schema_resource_name="execution.json", metadata=metadata, **kwargs)

        self._hash = codehash
        self._command = command

    def _execute(self, *args, parameter_map: PipelineParameters, **kwargs):
        from pypadre.core.model.computation.run import Run
        run = Run(execution=self)
        self.experiment.pipeline.execute(data=self.experiment.dataset, run=run, parameter_map=parameter_map, execution=self, **kwargs)

    @property
    def hash(self):
        return self._hash

    @property
    def command(self):
        return self._command

    @property
    def experiment(self):
        return self.parent
