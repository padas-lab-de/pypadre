from pypadre.core.model.split.split import Split
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.local.file.split_repository import SplitFileRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer

META_FILE = File("metadata.json", JSonSerializer)
RESULTS_FILE = File("results.json", JSonSerializer)
METRICS_FILE = File("metrics.json", JSonSerializer)


class SplitGitlabRepository(SplitFileRepository):

    def __init__(self, backend: IPadreBackend):
        super().__init__(backend=backend)
        self._gitlab_backend = self.backend.experiment

    def get(self, uid):
        return self._gitlab_backend.get(uid, rpath='executions/runs/splits', caller=self)

    def list(self, search, offset=0, size=100):
        if search is None:
            search = {self._gitlab_backend.RELATIVE_PATH: 'executions/runs/splits'}
        else:
            search[self._gitlab_backend.RELATIVE_PATH] = 'executions/runs/splits'
        return self._gitlab_backend.list(search, offset, size, caller=self)

    def _get_by_repo(self, repo, path=''):
        metadata = self._gitlab_backend.get_file(repo, META_FILE, path=path)

        split = Split(metadata=metadata)
        return split

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        super()._put(obj, *args, directory=directory, merge=merge, **kwargs)
        self.parent.update(obj.parent, commit_message="Creating a new split of the dataset")
