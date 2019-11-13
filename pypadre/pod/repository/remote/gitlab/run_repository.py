from cachetools import cached, LRUCache

from pypadre.core.model.computation.run import Run
from pypadre.core.util.utils import remove_cached
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.local.file.run_repository import RunFileRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer

META_FILE = File("metadata.json", JSonSerializer)

cache = LRUCache(maxsize=5)


class RunGitlabRepository(RunFileRepository):

    def __init__(self, backend: IPadreBackend):
        super().__init__(backend=backend)

    def list(self, search, offset=0, size=100):
        return self.backend.experiment.list(search, offset, size, caller=self)

    @cached(cache)
    def get(self, uid, rpath='executions/runs'):
        return self.backend.experiment.get(uid, rpath=rpath, caller=self)

    def _get_by_repo(self, repo, path=''):

        metadata = self.backend.experiment.get_file(repo, META_FILE, path=path)
        execution_path = '/'.join(path.split('/')[:-2])
        execution = self.parent._get_by_repo(repo,path=execution_path)
        run = Run(execution=execution, metadata=metadata)
        return run

    def update(self, run: Run, commit_message:str):
        self.parent.update(run.parent,commit_message=commit_message)

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        super()._put(obj,*args,directory=directory,merge=merge,**kwargs)
        self.parent.update(obj.parent,commit_message="Added a new run or updated an existing one to the experiment.")
        remove_cached(cache,obj.id)
