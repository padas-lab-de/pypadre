from pypadre.backend.interfaces.backend.i_metric_backend import IMetricBackend
from pypadre.backend.interfaces.backend.generic.i_base_file_backend import File
from pypadre.backend.serialiser import JSonSerializer


class PadreMetricFileBackend(IMetricBackend):
    def to_folder_name(self, obj):
        pass

    def get_by_dir(self, directory):
        pass

    METRICS_FILE = File("metrics.json", JSonSerializer)

    def __init__(self, parent):
        super().__init__(parent, name=parent.METRICS_FILE_NAME)

    def list(self, search, offset=0, size=100):
        pass

    def get(self, uid):
        return super().get()

    def put(self, metrics):

        directory = self.get_dir(self.to_folder_name(self.parent))
        self.write_file(directory, self.METRICS_FILE, metrics)

    def delete(self, uid):
        pass