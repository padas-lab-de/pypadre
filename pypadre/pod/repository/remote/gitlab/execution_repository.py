from pypadre.core.model.execution import Execution
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IExecutionRepository
from pypadre.pod.repository.local.file.execution_repository import ExecutionFileRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File, IChildFileRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer
from pypadre.pod.util.git_util import add_and_commit

NAME = 'executions'

# CONFIG_FILE = File("experiment.json", JSonSerializer)
META_FILE = File(NAME+ExecutionFileRepository.placeholder()+"/metadata.json", JSonSerializer)


class ExecutionGitlabRepository(ExecutionFileRepository):

    def __init__(self, backend: IPadreBackend):
        super().__init__(backend=backend)

    # def list(self, search, offset=0, size=100):
    #     return self.backend.experiment.list(search,offset,size)

    def get_by_repo(self,repo):
        #TODO repalce place_holder for the repo path
        metadata = self.backend.experiment.get_file(repo,META_FILE)
        experiment = self.backend.experiment.get(metadata.get("experiment_id"))
        return Execution(experiment=experiment, metadata=metadata)

    def update(self,execution: Execution, commit_message:str):
        self.parent.update(execution.parent,commit_message)

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        super()._put(obj,*args,directory=directory,merge=merge,**kwargs)
        self.parent.update(obj.parent, commit_message= "Added a new execution or updated existing one to the experiment.")
