from pypadre.core.model.computation.computation import Computation
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IRunRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File, IChildFileRepository
from pypadre.pod.repository.local.file.generic.i_log_file_repository import ILogFileRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer

NAME = "computations"

META_FILE = File("metadata.json", JSonSerializer)


class ComputationRepository(IChildFileRepository, ILogFileRepository, IRunRepository):

    @staticmethod
    def placeholder():
        return '{COMPUTATION_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(parent=backend.execution, name=NAME, backend=backend)

    def get_by_dir(self, directory):
        metadata = self.get_file(directory, META_FILE)

        # TODO Computation
        computation = Computation(metadata=metadata)
        return computation

    def put_progress(self, run, **kwargs):
        self.log(
            "RUN PROGRESS: {curr_value}/{limit}. phase={phase} \n".format(**kwargs))

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        computation = obj
        self.write_file(directory, META_FILE, computation.metadata)
