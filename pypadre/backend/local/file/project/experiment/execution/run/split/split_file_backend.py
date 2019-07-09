import os

from pypadre.backend.interfaces.backend.i_split_backend import ISplitBackend
from pypadre.backend.local.file.project.experiment.execution.run.split.result.result_file_backend import PadreResultFileBackend
from pypadre.backend.local.file.project.experiment.execution.run.split.metric.metric_file_backend import PadreMetricFileBackend


class PadreSplitFileBackend(ISplitBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "splits")
        self._result = PadreResultFileBackend(self)
        self._metric = PadreMetricFileBackend(self)

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