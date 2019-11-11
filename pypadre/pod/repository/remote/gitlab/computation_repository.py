from pypadre.core.model.computation.computation import Computation
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.local.file.computation_repository import ComputationFileRepository


class ComputationGitlabRepository(ComputationFileRepository):

    def __init__(self,backend: IPadreBackend):
        super().__init__(backend=backend)

    def _put(self, obj, *args, directory: str, store_results=False, merge=False, **kwargs):
        super()._put(obj,*args,directory=directory,store_results=store_results, merge=merge, **kwargs)
        self.parent.update(obj.parent, commit_message="Added/updated a component computation")