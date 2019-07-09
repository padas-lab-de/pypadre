from abc import abstractmethod, ABC


class IBackend(ABC):

    @property
    @abstractmethod
    def dataset(self):
        pass

    @property
    @abstractmethod
    def project(self):
        pass
