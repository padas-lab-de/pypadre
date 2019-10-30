from pypadre.core.model.computation.pipeline_output import PipelineOutput
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IPipelineOutputRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File, IChildFileRepository
from pypadre.pod.repository.local.file.generic.i_log_file_repository import ILogFileRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer

NAME = "output"

META_FILE = File("metadata.json", JSonSerializer)
PARAMETER_FILE = File("parameters.json", JSonSerializer)
METRIC_FILE = File("metrics.json", JSonSerializer)
RESULT_FILE = File("results.json", JSonSerializer)


class PipelineOutputFileRepository(IChildFileRepository, ILogFileRepository, IPipelineOutputRepository):

    @staticmethod
    def placeholder():
        return '{COMPUTATION_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(parent=backend.run, name=NAME, backend=backend)

    def get_by_dir(self, directory):
        metadata = self.get_file(directory, META_FILE)
        parameter = self.get_file(directory, PARAMETER_FILE)
        metric = self.get_file(directory, METRIC_FILE)
        result = self.get_file(directory, RESULT_FILE)

        # TODO PipelineOutput
        return PipelineOutput(parameter_selection=parameter, metrics=metric, results=result)

    def _put(self, obj, *args, directory: str, store_results=False, merge=False, **kwargs):
        self.write_file(directory, META_FILE, obj.metadata)
        self.write_file(directory, PARAMETER_FILE, obj.parameter_selection)
        self.write_file(directory, METRIC_FILE, obj.metrics)
        self.write_file(directory, RESULT_FILE, obj.results)
