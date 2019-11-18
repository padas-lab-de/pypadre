from pypadre.core.metrics.metrics import Metric
from pypadre.core.util.utils import remove_cached
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.local.file.metric_repository import MetricFileRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer

META_FILE = File("metadata.json", JSonSerializer)
RESULT_FILE = File("results.json", JSonSerializer)


class MetricGitlabRepository(MetricFileRepository):

    def __init__(self, backend: IPadreBackend):
        super().__init__(backend=backend)
        self._gitlab_backend = self.backend.experiment

    def get(self, uid):
        return self._gitlab_backend.get(uid, rpath='executions/runs/metrics', caller=self)

    def list(self, search, offset=0, size=100):
        if search is None:
            search = {self._gitlab_backend.RELATIVE_PATH: 'executions/runs/metrics'}
        else:
            search[self._gitlab_backend.RELATIVE_PATH] = 'executions/runs/metrics'
        return self.backend.experiment.list(search, offset, size, caller=self)

    def _get_by_repo(self, repo, path=''):
        metadata = self._gitlab_backend.get_file(repo, META_FILE, path=path)
        result = self._gitlab_backend.get_file(repo, RESULT_FILE, path=path)
        computation = self.backend.computation.get(metadata.get(Metric.COMPUTATION_ID))

        metric = Metric(name=metadata.get(Metric.NAME), computation=computation, metadata=metadata, result=result)
        return metric

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        super()._put(obj, *args, directory=directory, merge=merge, **kwargs)
        self.parent.update(obj.parent, commit_message="Adding new metrics results and metadata")
