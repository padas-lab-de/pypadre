import os

from pypadre.backend.interfaces.backend.i_project_backend import IProjectBackend
from pypadre.backend.local.file.project.experiment.experiment_file_backend import PadreExperimentFileBackend


class PadreProjectHTTPBackend(IProjectBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "projects")
        self._experiment = PadreExperimentFileBackend(self)

    @property
    def experiment(self):
        return self._experiment

    def list(self, search):
        pass

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass