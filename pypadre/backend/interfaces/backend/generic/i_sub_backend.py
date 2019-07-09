from abc import abstractmethod, ABCMeta


class ISubBackend:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, parent):
        self._parent = parent
