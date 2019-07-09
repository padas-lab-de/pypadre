from abc import abstractmethod, ABCMeta


class ISearchable:
    __metaclass__ = ABCMeta

    @abstractmethod
    def list(self, search: dict):
        pass
