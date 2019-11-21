from pypadre.core.model.execution import Execution
from pypadre.core.model.generic.custom_code import CodeManagedMixin
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.local.file.execution_repository import ExecutionFileRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.serializer.serialiser import JSonSerializer

NAME = 'executions'

# CONFIG_FILE = File("experiment.json", JSonSerializer)
META_FILE = File("metadata.json", JSonSerializer)


class ExecutionGitlabRepository(ExecutionFileRepository):

    def __init__(self, backend: IPadreBackend):
        super().__init__(backend=backend)
        self._gitlab_backend = self.backend.experiment

    def list(self, search, offset=0, size=100):
        if search is None:
            search = {self._gitlab_backend.RELATIVE_PATH: 'executions'}
        else:
            search[self._gitlab_backend.RELATIVE_PATH] = 'executions'
        return self._gitlab_backend.list(search, offset, size, caller=self)

    def get(self, uid, rpath=NAME):
        return self._gitlab_backend.get(uid, rpath=rpath, caller=self)

    def _get_by_repo(self, repo, path=''):
        metadata = self._gitlab_backend.get_file(repo, META_FILE, path=path)
        experiment = self.parent._get_by_repo(repo, path='')
        reference = self.backend.code.get(metadata.get(CodeManagedMixin.DEFINED_IN))

        return Execution(experiment=experiment, metadata=metadata, reference=reference)

    def update(self, execution: Execution, commit_message: str):
        self.parent.update(execution.parent, commit_message)

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        super()._put(obj, *args, directory=directory, merge=merge, **kwargs)
        self.parent.update(obj.parent,
                           commit_message="Added a new execution or updated existing one to the experiment.")
