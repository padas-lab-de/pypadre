from types import GeneratorType

from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.generic.lazy_loader import SimpleLazyObject
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IComputationRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File, IChildFileRepository
from pypadre.pod.repository.local.file.generic.i_log_file_repository import ILogFileRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer, DillSerializer

NAME = "computations"

META_FILE = File("metadata.json", JSonSerializer)
PARAMETER_FILE = File("parameters.json", JSonSerializer)
RESULT_FILE = File("results.bin", DillSerializer)
INITIAL_HYPERPARAMETERS = File("initial_hyperparameters.json", JSonSerializer)


class ComputationFileRepository(IChildFileRepository, ILogFileRepository, IComputationRepository):

    @staticmethod
    def placeholder():
        return '{COMPUTATION_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(parent=backend.run, name=NAME, backend=backend)

    def get_by_dir(self, directory):
        metadata = self.get_file(directory, META_FILE)
        result = self.get_file(directory, RESULT_FILE)

        # TODO Computation
        run = self.backend.run.get(metadata.get(Computation.RUN_ID))
        component = run.pipeline.get_component(metadata.get(Computation.COMPONENT_ID))
        predecessor = None
        if metadata.get(Computation.PREDECESSOR_ID) is not None:
            predecessor = SimpleLazyObject(load_fn=lambda b: self.get(metadata.get(Computation.PREDECESSOR_ID)), id=metadata.get(Computation.PREDECESSOR_ID), clz=Computation)

        computation = Computation(metadata=metadata, result=result, run=run, component=component, predecessor=predecessor)
        return computation

    def put_progress(self, run, **kwargs):
        self.log(
            "RUN COMPUTATION: {curr_value}/{limit}. phase={phase} \n".format(**kwargs))

    def _put(self, obj, *args, directory: str, store_results=False, merge=False, **kwargs):
        computation = obj
        self.write_file(directory, META_FILE, computation.metadata)
        self.write_file(directory, PARAMETER_FILE, computation.parameters)
        self.write_file(directory, INITIAL_HYPERPARAMETERS, computation.initial_hyperparameters)
        if not isinstance(computation.result, GeneratorType) and store_results:
            self.write_file(directory, RESULT_FILE, computation.result, mode='wb')
