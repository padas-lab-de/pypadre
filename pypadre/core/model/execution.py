from pypadre.core.base import MetadataEntity
from pypadre.core.events.events import signals
from pypadre.core.model.generic.i_model_mixins import IStoreable, IProgressable, IExecuteable
from pypadre.core.printing.tablefyable import Tablefyable


class Execution(IStoreable, IProgressable, IExecuteable, MetadataEntity, Tablefyable):
    """ A execution should save data about the running env and the version of the code on which it was run """

    def _execute(self, *args, **kwargs):
        from pypadre.core.model.computation.run import Run
        run = Run(execution=self)
        self.experiment.pipeline.execute(run=run, data=self.experiment.dataset, execution=self, **kwargs)

    @classmethod
    def _tablefy_register_columns(cls):
        # Add entries for tablefyable
        cls.tablefy_register_columns({'hash': 'hash', 'cmd': 'cmd'})

    def __init__(self, experiment, codehash=None, command=None, **kwargs):
        metadata = {"id": codehash, **kwargs, "command": command, "codehash": codehash}
        super().__init__(schema_resource_name="execution.json", metadata=metadata, **kwargs)

        self._experiment = experiment
        self._hash = codehash
        self._command = command

    @property
    def hash(self):
        return self._hash

    @property
    def command(self):
        return self._command

    @property
    def experiment(self):
        return self._experiment
