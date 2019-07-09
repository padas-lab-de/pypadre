import os

from pypadre.backend.interfaces.backend.i_split_backend import ISplitBackend
from pypadre.backend.remote.http.project.experiment.execution.run.split.result.result_http_backend import PadreResultHTTPBackend
from pypadre.backend.remote.http.project.experiment.execution.run.split.metric.metric_http_backend import PadreMetricHTTPBackend


class PadreSplitHTTPBackend(ISplitBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "splits")
        self._result = PadreResultHTTPBackend(self)
        self._metric = PadreMetricHTTPBackend(self)

    @property
    def result(self):
        return self._result

    @property
    def metric(self):
        return self._metric

    def put_progress(self, obj):pass

    def list(self, search):
        pass

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass