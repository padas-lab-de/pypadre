import os

from pypadre.backend.interfaces.backend.i_split_backend import ISplitBackend
from pypadre.backend.local.file.project.experiment.execution.run.split.split_file_backend import PadreSplitFileBackend
from pypadre.backend.remote.http.project.experiment.execution.run.split.result.result_http_backend import PadreResultHttpBackend
from pypadre.backend.remote.http.project.experiment.execution.run.split.metric.metric_http_backend import PadreMetricHttpBackend


class PadreSplitHttpBackend(PadreSplitFileBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "splits")
        self._result = PadreResultHttpBackend(self)
        self._metric = PadreMetricHttpBackend(self)

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