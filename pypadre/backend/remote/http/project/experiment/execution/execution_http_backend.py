import os

from pypadre.backend.interfaces.backend.i_execution_backend import IExecutionBackend
from pypadre.backend.local.file.project.experiment.execution.execution_file_backend import PadreExecutionFileBackend
from pypadre.backend.remote.http.project.experiment.execution.run.run_http_backend import PadreRunHttpBackend


class PadreExecutionHttpBackend(PadreExecutionFileBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "executions")
        self._run = PadreRunHttpBackend(self)

    @property
    def run(self):
        return self._run

    def list(self, search, offset=0, size=100):
        pass

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass