from pypadre.core.model.split.split import Split
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import ISplitRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File, IChildFileRepository
from pypadre.pod.repository.local.file.generic.i_log_file_repository import ILogFileRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer

NAME = "splits"

META_FILE = File("metadata.json", JSonSerializer)
RESULTS_FILE = File("results.json", JSonSerializer)
METRICS_FILE = File("metrics.json", JSonSerializer)


class SplitFileRepository(IChildFileRepository, ILogFileRepository, ISplitRepository):

    @staticmethod
    def placeholder():
        return '{SPLIT_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(parent=backend.run, name=NAME, backend=backend)

    def _get_by_dir(self, directory):
        metadata = self.get_file(directory, META_FILE)

        # TODO Computation
        split = Split(metadata=metadata)
        return split

    def put_progress(self, run, **kwargs):
        self.log(
            "SPLIT PROGRESS: {curr_value}/{limit}. phase={phase} \n".format(**kwargs))

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        computation = obj
        self.write_file(directory, META_FILE, computation.metadata)
