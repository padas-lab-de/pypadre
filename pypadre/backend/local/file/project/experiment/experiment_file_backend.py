import os

from pypadre.backend.interfaces.backend.i_experiment_backend import IExperimentBackend
from pypadre.backend.local.file.project.experiment.execution.execution_file_backend import PadreExecutionFileBackend


class PadreExperimentFileBackend(IExperimentBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "experiments")
        self._execution = PadreExecutionFileBackend(self)

    @property
    def execution(self):
        pass

    def log(self, msg):
        pass

    def put_config(self, obj):
        pass

    def put_progress(self, obj):
        pass

    def list(self, search):
        pass

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass