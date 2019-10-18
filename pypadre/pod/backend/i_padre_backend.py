from abc import abstractmethod

from pypadre.pod.repository.generic.i_repository_mixins import ILogRepository
from pypadre.pod.repository.i_repository import IComputationRepository, IMetricRepository, ICodeRepository


class IPadreBackend(ILogRepository):
    """ This is the base backend for padre. It contains subbackends like dataset and project."""

    from pypadre.pod.repository.i_repository import IProjectRepository, IDatasetRepository, IExperimentRepository, \
        ISplitRepository, IRunRepository, IExecutionRepository

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self._config = config

    @property
    def config(self):
        return self._config

    @property
    def root_dir(self):
        return self._config.get("root_dir")

    @property
    @abstractmethod
    def dataset(self) -> IDatasetRepository:
        raise NotImplementedError()

    @property
    @abstractmethod
    def project(self) -> IProjectRepository:
        raise NotImplementedError()

    @property
    @abstractmethod
    def experiment(self) -> IExperimentRepository:
        raise NotImplementedError()

    @property
    @abstractmethod
    def execution(self) -> IExecutionRepository:
        raise NotImplementedError()

    @property
    @abstractmethod
    def run(self) -> IRunRepository:
        raise NotImplementedError()

    @property
    @abstractmethod
    def split(self) -> ISplitRepository:
        raise NotImplementedError()

    @property
    @abstractmethod
    def computation(self) -> IComputationRepository:
        raise NotImplementedError()

    @property
    @abstractmethod
    def metric(self) -> IMetricRepository:
        raise NotImplementedError()

    @property
    def code(self) -> ICodeRepository:
        raise NotImplementedError()

