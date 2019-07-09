import os

from pypadre.backend.interfaces.backend.i_run_backend import IRunBackend
from pypadre.backend.remote.http.project.experiment.execution.run.split.split_http_backend import PadreSplitHTTPBackend


class PadreRunHTTPBackend(IRunBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "runs")
        self._split = PadreSplitHTTPBackend(self)

    @property
    def split(self):
        return self._split

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