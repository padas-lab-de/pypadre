from pypadre.core.model.execution import Execution
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IExecutionRepository
from pypadre.pod.repository.local.file.execution_repository import ExecutionFileRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File, IChildFileRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer
from pypadre.pod.util.git_util import add_and_commit

NAME = 'executions'

# CONFIG_FILE = File("experiment.json", JSonSerializer)
META_FILE = File("metadata.json", JSonSerializer)


class ExecutionGitlabRepository(ExecutionFileRepository):


    def __init__(self, backend: IPadreBackend):
        super().__init__(backend=backend)


    def get(self, uid):
        """
        Shortcut because we know the uid is the folder name
        :param uid: Uid of the execution
        :return:
        """
        # TODO: Execution folder name is the hash. Get by uid will require looking into the metadata
        return super().get(uid)

    def update(self,obj, commit_message:str):
        self.parent.update(obj,commit_message)

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        super()._put(obj,*args,directory,merge,**kwargs)
        self.parent.update(obj.experiment, commit_message= "Added a new execution or updated existing one to the experiment.")
        # The code for each execution changes. So it is necessary to write the experiment.json file too.
        # self.write_file(directory, CONFIG_FILE, execution.config)
