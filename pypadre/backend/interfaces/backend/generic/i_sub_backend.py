from abc import abstractmethod, ABCMeta


class ISubBackend:
    """ This is the abstract class of a backend being hierarchically nested in another backend. For example the
    project backend is the parent of experiment backend"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, parent, **kwargs):
        self._parent = parent
