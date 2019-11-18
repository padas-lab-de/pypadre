from pypadre.core.model.computation.pipeline_output import PipelineOutput
from pypadre.core.util.utils import remove_cached
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.local.file.pipeline_output_repository import PipelineOutputFileRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer

META_FILE = File("metadata.json", JSonSerializer)
PARAMETER_FILE = File("parameters.json", JSonSerializer)
METRIC_FILE = File("metrics.json", JSonSerializer)
RESULT_FILE = File("results.json", JSonSerializer)


class PipelineOutputGitlabRepository(PipelineOutputFileRepository):

    def __init__(self, backend: IPadreBackend):
        super().__init__(backend=backend)
        self._gitlab_backend = self.backend.experiment

    def get(self, uid):
        return self.backend.experiment.get(uid, rpath='executions/runs/output', caller=self)

    def list(self, search, offset=0, size=100):
        if search is None:
            search = {self._gitlab_backend.RELATIVE_PATH: 'executions/runs/output'}
        else:
            search[self._gitlab_backend.RELATIVE_PATH] = 'executions/runs/output'
        return self._gitlab_backend.list(search, offset, size, caller=self)

    def _get_by_repo(self, repo, path=''):

        metadata = self.backend.experiment.get_file(repo, META_FILE, path=path)
        parameter = self.backend.experiment.get_file(repo, PARAMETER_FILE, {}, path=path)
        metric = self.backend.experiment.get_file(repo, METRIC_FILE, path=path)
        result = self.backend.experiment.get_file(repo, RESULT_FILE, path=path)

        run_path = '/'.join(path.split('/')[:-2])
        run = self.parent._get_by_repo(repo, path=run_path)

        splits = set()
        for split_id in metadata.get(PipelineOutput.SPLIT_IDS, []):
            splits.add(self.backend.computation.get(split_id))

        # TODO PipelineOutput
        return PipelineOutput(run=run, splits=splits, parameter_selection=parameter, metrics=metric, results=result,
                              metadata=metadata)

    def _put(self, obj, *args, directory: str, store_results=False, merge=False, **kwargs):
        super()._put(obj, *args, directory=directory, store_results=store_results, merge=merge, **kwargs)
        self.parent.update(obj.parent,
                           commit_message="Added metadata, parameter selection, metrics and results of the whole pipeline")
