from abc import abstractmethod

from pypadre.pod.repository.generic.i_repository_mixins import ILogRepository
from pypadre.pod.repository.i_repository import IComputationRepository


class IPadreBackend(ILogRepository):
    """ This is the base backend for padre. It contains subbackends like dataset and project."""

    from pypadre.pod.repository.i_repository import IProjectRepository, IDatasetRepository, IExperimentRepository, \
        ISplitRepository, IRunRepository, IExecutionRepository

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self._config = config

        # TODO Receiver?
        def handle_put(sender, **kwargs):
            pass

    @property
    def config(self):
        return self._config

    @property
    def root_dir(self):
        return self._config.get("root_dir")

    @property
    @abstractmethod
    def dataset(self) -> IDatasetRepository:
        pass

    @property
    @abstractmethod
    def project(self) -> IProjectRepository:
        pass

    @property
    @abstractmethod
    def experiment(self) -> IExperimentRepository:
        pass

    @property
    @abstractmethod
    def execution(self) -> IExecutionRepository:
        pass

    @property
    @abstractmethod
    def run(self) -> IRunRepository:
        pass

    @property
    @abstractmethod
    def split(self) -> ISplitRepository:
        pass

    @property
    @abstractmethod
    def computation(self) -> IComputationRepository:
        pass
