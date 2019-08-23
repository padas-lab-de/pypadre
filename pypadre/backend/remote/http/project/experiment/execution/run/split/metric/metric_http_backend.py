from pypadre.backend.interfaces.backend.i_metric_backend import IMetricBackend
from pypadre.backend.local.file.project.experiment.execution.run.split.metric.metric_file_backend import \
    PadreMetricFileBackend


class PadreMetricHttpBackend(PadreMetricFileBackend):

    def to_folder_name(self, obj):
        pass

    def get_by_dir(self, directory):
        pass

    def __init__(self, parent):
        super().__init__(parent)

    def list(self, search, offset=0, size=100):
        pass

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass