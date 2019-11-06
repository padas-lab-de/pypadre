import os

from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IProjectRepository, IExperimentRepository, IDatasetRepository, \
    IComputationRepository, IMetricRepository, ICodeRepository, IPipelineOutputRepository
from pypadre.pod.repository.local.file.computation_repository import ComputationFileRepository
from pypadre.pod.repository.local.file.metric_repository import MetricFileRepository
from pypadre.pod.repository.local.file.pipeline_output_repository import PipelineOutputFileRepository
from pypadre.pod.repository.local.file.run_repository import RunFileRepository
from pypadre.pod.repository.local.file.split_repository import SplitFileRepository
from pypadre.pod.repository.remote.gitlab.code_repository import CodeGitlabRepository
from pypadre.pod.repository.remote.gitlab.dataset_repository import DatasetGitlabRepository
from pypadre.pod.repository.remote.gitlab.execution_repository import ExecutionGitlabRepository
from pypadre.pod.repository.remote.gitlab.experiment_repository import ExperimentGitlabRepository
from pypadre.pod.repository.remote.gitlab.project_repository import ProjectGitlabRepository


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
        # if self._file is None:
        #     self._file = open(os.path.join(self.root_dir, "padre.log"), "a")
        # self._file.write(message)
        pass

    def __init__(self, config):
        super().__init__(config)
        #TODO finsh all backends repos
        self._project = ProjectGitlabRepository(self)
        self._experiment = ExperimentGitlabRepository(self)
        self._dataset = DatasetGitlabRepository(self)
        self._execution = ExecutionGitlabRepository(self)
        self._run = RunFileRepository(self)
        self._split = SplitFileRepository(self)
        self._computation = ComputationFileRepository(self)
        self._metric = MetricFileRepository(self)
        self._code = CodeGitlabRepository(self)
        self._pipeline_output = PipelineOutputFileRepository(self)

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
    def run(self) -> RunFileRepository:
        return self._run

    @property
    def split(self) -> SplitFileRepository:
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