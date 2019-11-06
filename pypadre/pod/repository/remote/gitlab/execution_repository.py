from pypadre.core.model.execution import Execution
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IExecutionRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File, IChildFileRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer
from pypadre.pod.util.git_util import add_and_commit

NAME = 'executions'

# CONFIG_FILE = File("experiment.json", JSonSerializer)
META_FILE = File("metadata.json", JSonSerializer)


class ExecutionGitlabRepository(IChildFileRepository, IExecutionRepository):

    @staticmethod
    def placeholder():
        return '{EXECUTION_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(parent=backend.experiment, name=NAME, backend=backend)

    def to_folder_name(self, execution):
        return str(execution.hash)

    def get(self, uid):
        """
        Shortcut because we know the uid is the folder name
        :param uid: Uid of the execution
        :return:
        """
        # TODO: Execution folder name is the hash. Get by uid will require looking into the metadata
        return super().get(uid)

    def get_by_dir(self, directory):
        metadata = self.get_file(directory, META_FILE)
        experiment = self.backend.experiment.get(metadata.get("experiment_id"))
        return Execution(experiment=experiment, metadata=metadata)

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        execution = obj
        self.write_file(directory, META_FILE, execution.metadata)
        add_and_commit(self.parent.root_dir)
        if self.parent.remote is not None:
            self.parent.push_changes()
        # The code for each execution changes. So it is necessary to write the experiment.json file too.
        # self.write_file(directory, CONFIG_FILE, execution.config)
