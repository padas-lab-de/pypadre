from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.local.file.generic.i_file_repository import File, IChildFileRepository
from pypadre.pod.repository.i_repository import IExecutionRepository
from pypadre.core.model.execution import Execution
from pypadre.pod.repository.serializer.serialiser import JSonSerializer

NAME = 'executions'

CONFIG_FILE = File("experiment.json", JSonSerializer)
META_FILE = File("metadata.json", JSonSerializer)


class ExecutionFileRepository(IChildFileRepository, IExecutionRepository):

    @staticmethod
    def placeholder():
        return '{EXECUTION_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(parent=backend.experiment, name=NAME, backend=backend)

    def to_folder_name(self, execution):
        return execution.name

    def get(self, uid):
        """
        Shortcut because we know the uid is the folder name
        :param uid: Uid of the execution
        :return:
        """
        # TODO might be changed. Execution get folder name or id by git commit hash?
        return super().get(uid)

    def get_by_dir(self, directory):
        metadata = self.get_file(directory, META_FILE)
        experiment = self.backend.experiment.get(metadata.get("experiment_id"))
        return Execution(experiment=experiment, **metadata)

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        execution = obj
        self.write_file(directory, META_FILE, execution.metadata)

        # The code for each execution changes. So it is necessary to write the experiment.json file too.
        self.write_file(directory, CONFIG_FILE, execution.config)
