import os

from pypadre.backend.interfaces.backend.i_experiment_backend import IExperimentBackend
from pypadre.backend.remote.http.project.experiment.execution.execution_http_backend import PadreExecutionHTTPBackend


class PadreExperimentHTTPBackend(IExperimentBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "experiments")
        self._execution = PadreExecutionHTTPBackend(self)

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