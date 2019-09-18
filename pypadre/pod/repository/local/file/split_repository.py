from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.local.file.generic.i_file_repository import File, IChildFileRepository
from pypadre.pod.repository.local.file.generic.i_log_file_repository import ILogFileRepository
from pypadre.pod.repository.i_repository import ISplitRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer

NAME = "splits"

RESULTS_FILE = File("results.json", JSonSerializer)
METRICS_FILE = File("metrics.json", JSonSerializer)
METADATA_FILE = File("metadata.json", JSonSerializer)


class SplitFileRepository(IChildFileRepository, ILogFileRepository, ISplitRepository):

    @staticmethod
    def placeholder():
        return '{SPLIT_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(parent=backend.run, name=NAME, backend=backend)

    @property
    def result(self):
        # TODO
        return self._result

    @property
    def metric(self):
        # TODO
        return self._metric

    def put_progress(self, split, **kwargs):
        self.log(
            "Split " + split + " PROGRESS: {curr_value}/{limit}. phase={phase} \n".format(**kwargs))

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        split = obj
        self.write_file(directory, METADATA_FILE, split.metadata)

    def get_by_dir(self, directory):
        pass
