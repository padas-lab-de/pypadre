from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.local.file.pipeline_output_repository import PipelineOutputFileRepository


class PipelineOutputGitlabRepository(PipelineOutputFileRepository):

    def __init__(self, backend: IPadreBackend):
        super().__init__(backend=backend)

    def list(self, search, offset=0, size=100):
        return self.backend.experiment.list(search,offset,size)

    def get_by_repo(self,repo):
        #TODO
        pass

    def _put(self, obj, *args, directory: str, store_results=False, merge=False, **kwargs):
        super()._put(obj, *args, directory=directory, store_results=store_results, merge=merge, **kwargs)
        self.parent.update(obj.parent, commit_message="Added metadata, parameter selection, metrics and results of the whole pipeline")