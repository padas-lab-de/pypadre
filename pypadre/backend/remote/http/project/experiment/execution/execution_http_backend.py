import os

from pypadre.backend.interfaces.backend.i_execution_backend import IExecutionBackend
from pypadre.backend.remote.http.project.experiment.execution.run.run_http_backend import PadreRunHTTPBackend


class PadreExecutionHTTPBackend(IExecutionBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "executions")
        self._run = PadreRunHTTPBackend(self)

    @property
    def run(self):
        return self._run

    def list(self, search):
        pass

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass