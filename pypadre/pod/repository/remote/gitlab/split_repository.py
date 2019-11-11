from pypadre.core.model.split.split import Split
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.local.file.split_repository import SplitFileRepository


class SplitGitlabRepository(SplitFileRepository):

    def __init__(self, backend: IPadreBackend):
        super().__init__(backend=backend)

    def get_by_repo(self,repo):
        #TODO
        pass

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        super()._put(obj,*args, directory=directory,merge=merge,**kwargs)
        self.parent.update(obj.parent,commit_message = "Creating a new split of the dataset")