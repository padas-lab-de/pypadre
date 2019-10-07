from types import GeneratorType

from pypadre.core.model.computation.computation import Computation
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IRunRepository, IComputationRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File, IChildFileRepository
from pypadre.pod.repository.local.file.generic.i_log_file_repository import ILogFileRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer, PickleSerializer

NAME = "computations"

META_FILE = File("metadata.json", JSonSerializer)
RESULT_FILE = File("results.bin", PickleSerializer)


class ComputationFileRepository(IChildFileRepository, ILogFileRepository, IComputationRepository):

    @staticmethod
    def placeholder():
        return '{COMPUTATION_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(parent=backend.execution, name=NAME, backend=backend)

    def get_by_dir(self, directory):
        metadata = self.get_file(directory, META_FILE)
        result = self.get_file(directory, RESULT_FILE)

        # TODO Computation
        computation = Computation(metadata=metadata, result=result)
        return computation

    def put_progress(self, run, **kwargs):
        self.log(
            "RUN COMPUTATION: {curr_value}/{limit}. phase={phase} \n".format(**kwargs))

    def _put(self, obj, *args, directory: str, store_results=False, merge=False, **kwargs):
        computation = obj
        self.write_file(directory, META_FILE, computation.metadata)
        if not isinstance(computation.result, GeneratorType) and store_results:
            self.write_file(directory, RESULT_FILE, computation.result, mode='wb')
