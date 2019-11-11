from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.local.file.metric_repository import MetricFileRepository


class MetricGitlabRepository(MetricFileRepository):

    def __init__(self, backend: IPadreBackend):
        super().__init__(backend=backend)

    def get_by_repo(self,repo):
        #Todo
        pass

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        super()._put(obj, *args, directory=directory, merge=merge, **kwargs)
        self.parent.update(obj.parent,commit_message="Adding new metrics results and metadata")