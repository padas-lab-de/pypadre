from pypadre.core.model.computation.computation import Computation
from pypadre.core.model.generic.lazy_loader import SimpleLazyObject
from pypadre.core.util.utils import remove_cached
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.local.file.computation_repository import ComputationFileRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.serializer.serialiser import JSonSerializer, DillSerializer

META_FILE = File("metadata.json", JSonSerializer)
PARAMETER_FILE = File("parameters.json", JSonSerializer)
RESULT_FILE = File("results.bin", DillSerializer)
INITIAL_HYPERPARAMETERS = File("initial_hyperparameters.json", JSonSerializer)


class ComputationGitlabRepository(ComputationFileRepository):

    def __init__(self, backend: IPadreBackend):
        super().__init__(backend=backend)
        self._gitlab_backend = self.backend.experiment

    def get(self, uid):
        return self.backend.experiment.get(uid, rpath='executions/runs/computations', caller=self)

    def list(self, search, offset=0, size=100):
        if search is None:
            search = {self._gitlab_backend.RELATIVE_PATH: 'executions/runs/computations'}
        else:
            search[self._gitlab_backend.RELATIVE_PATH] = 'executions/runs/computations'
        return self._gitlab_backend.list(search, offset, size, caller=self)

    def _get_by_repo(self, repo, path=''):
        metadata = self._gitlab_backend.get_file(repo, META_FILE, path=path)
        result = self._gitlab_backend.get_file(repo, RESULT_FILE, path=path)
        parameters = self._gitlab_backend.get_file(repo, PARAMETER_FILE, default={}, path=path)

        # TODO Computation
        run_path = '/'.join(path.split('/')[:-2])
        run = self.parent._get_by_repo(repo, path=run_path)
        component = run.pipeline.get_component(metadata.get(Computation.COMPONENT_ID))
        predecessor = None
        if metadata.get(Computation.PREDECESSOR_ID) is not None:
            predecessor = SimpleLazyObject(load_fn=lambda b: self.get(metadata.get(Computation.PREDECESSOR_ID)),
                                           id=metadata.get(Computation.PREDECESSOR_ID), clz=Computation)

        computation = Computation(metadata=metadata, parameters=parameters, result=result, run=run, component=component,
                                  predecessor=predecessor)
        return computation

    def _put(self, obj, *args, directory: str, store_results=False, merge=False, **kwargs):
        super()._put(obj, *args, directory=directory, store_results=store_results, merge=merge, **kwargs)
        self.parent.update(obj.parent, commit_message="Added/updated a component computation")
