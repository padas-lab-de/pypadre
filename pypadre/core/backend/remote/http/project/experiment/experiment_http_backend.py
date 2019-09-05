import os

from pypadre.core.backend.local.file.project.experiment.experiment_file_backend import PadreExperimentFileBackend
from pypadre.core.backend.remote.http.project.experiment.execution.execution_http_backend import \
    PadreExecutionHttpBackend


class PadreExperimentHttpBackend(PadreExperimentFileBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "experiments")
        self._execution = PadreExecutionHttpBackend(self)

    @property
    def execution(self):
        pass

    def log(self, msg):
        pass

    def put_config(self, obj):
        pass

    def put_progress(self, obj):
        pass

    def list(self, search, offset=0, size=100):
        pass

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass