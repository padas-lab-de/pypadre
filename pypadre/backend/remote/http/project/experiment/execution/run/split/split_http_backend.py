import os

from pypadre.backend.interfaces.backend.i_split_backend import ISplitBackend
from pypadre.backend.local.file.project.experiment.execution.run.split.split_file_backend import PadreSplitFileBackend


class PadreSplitHttpBackend(PadreSplitFileBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "splits")

    @property
    def result(self):
        return self._result

    @property
    def metric(self):
        return self._metric

    def put_progress(self, obj):pass

    def list(self, search, offset=0, size=100):
        pass

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass