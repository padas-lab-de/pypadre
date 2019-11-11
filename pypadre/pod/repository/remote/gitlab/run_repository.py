from pypadre.core.model.computation.run import Run
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.local.file.run_repository import RunFileRepository


class RunGitlabRepository(RunFileRepository):

    def __init__(self, backend: IPadreBackend):
        super().__init__(backend=backend)

    # def list(self, search, offset=0, size=100):
    #     return self.backend.experiment.list(search,offset,size)

    def get_by_repo(self, repo):
        #TODO
        return

    def update(self, run: Run, commit_message:str):
        self.parent.update(run.parent,commit_message=commit_message)

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        super()._put(obj,*args,directory=directory,merge=merge,**kwargs)
        self.parent.update(obj.parent,commit_message="Added a new run or updated an existing one to the experiment.")