import os

from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IProjectRepository, IExperimentRepository, IDatasetRepository, \
    IComputationRepository, IMetricRepository, ICodeRepository, IPipelineOutputRepository
from pypadre.pod.repository.remote.gitlab.code_repository import CodeGitlabRepository
from pypadre.pod.repository.remote.gitlab.computation_repository import ComputationGitlabRepository
from pypadre.pod.repository.remote.gitlab.dataset_repository import DatasetGitlabRepository
from pypadre.pod.repository.remote.gitlab.execution_repository import ExecutionGitlabRepository
from pypadre.pod.repository.remote.gitlab.experiment_repository import ExperimentGitlabRepository
from pypadre.pod.repository.remote.gitlab.metric_repository import MetricGitlabRepository
from pypadre.pod.repository.remote.gitlab.pipeline_output_repository import PipelineOutputGitlabRepository
from pypadre.pod.repository.remote.gitlab.project_repository import ProjectGitlabRepository
from pypadre.pod.repository.remote.gitlab.run_repository import RunGitlabRepository
from pypadre.pod.repository.remote.gitlab.split_repository import SplitGitlabRepository


class PadreGitLabBackend(IPadreBackend):
    """
    backend class holding the gitlab repositories for our padre objects
    """
    def log_info(self, message, **kwargs):
        self.log(message="INFO: " + message, **kwargs)

    def log_warn(self, message, **kwargs):
        self.log(message="WARN: " + message, **kwargs)

    def log_error(self, message, **kwargs):
        self.log(message="ERROR: " + message, **kwargs)

    def log(self, message, **kwargs):
        if self._file is None:
            path = os.path.join(self.root_dir, "padre.log")
            if not os.path.exists(self.root_dir):
                os.makedirs(self.root_dir)

            self._file = open(path, "a")

        self._file.write(message)

    def __init__(self, config):
        super().__init__(config)
        #TODO finsh all backends repos
        self._project = ProjectGitlabRepository(self)
        self._experiment = ExperimentGitlabRepository(self)
        self._dataset = DatasetGitlabRepository(self)
        self._execution = ExecutionGitlabRepository(self)
        self._run = RunGitlabRepository(self)
        self._split = SplitGitlabRepository(self)
        self._computation = ComputationGitlabRepository(self)
        self._metric = MetricGitlabRepository(self)
        self._code = CodeGitlabRepository(self)
        self._pipeline_output = PipelineOutputGitlabRepository(self)

        # logging
        self._file = None

    @property
    def project(self) -> IProjectRepository:
        return self._project

    @property
    def experiment(self) -> IExperimentRepository:
        return self._experiment

    @property
    def dataset(self) -> IDatasetRepository:
        return self._dataset

    @property
    def execution(self) -> ExecutionGitlabRepository:
        return self._execution

    @property
    def run(self) -> RunGitlabRepository:
        return self._run

    @property
    def split(self) -> SplitGitlabRepository:
        return self._split

    @property
    def computation(self) -> IComputationRepository:
        return self._computation

    @property
    def metric(self) -> IMetricRepository:
        return self._metric

    @property
    def code(self) -> ICodeRepository:
        return self._code

    @property
    def pipeline_output(self) -> IPipelineOutputRepository:
        return self._pipeline_output