import os

from pypadre.pod.backend.local.file.project.project_file_backend import PadreProjectFileBackend
from pypadre.pod.backend.remote.http.project.experiment.experiment_http_backend import PadreExperimentHttpBackend


class PadreProjectHttpBackend(PadreProjectFileBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "projects")
        self._experiment = PadreExperimentHttpBackend(self)

    @property
    def experiment(self):
        return self._experiment

    def list(self, search, offset=0, size=100):
        pass

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass