from abc import abstractmethod

from pypadre.pod.backend.interfaces.backend.i_backend import IBackend
from pypadre.pod.backend.interfaces.backend.i_dataset_backend import IDatasetBackend
from pypadre.pod.backend.interfaces.backend.i_project_backend import IProjectBackend


class IPadreBackend(IBackend):
    """ This is the base backend for padre. It contains subbackends like dataset and project."""

    def __init__(self, config):
        super().__init__(config)

    @property
    @abstractmethod
    def dataset(self) -> IDatasetBackend:
        pass

    @property
    @abstractmethod
    def project(self) -> IProjectBackend:
        pass

