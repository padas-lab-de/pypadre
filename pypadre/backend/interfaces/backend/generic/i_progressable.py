from abc import abstractmethod, ABCMeta


class IProgressable:
    """ This is the interface for all backends being able to progress the state of one of their
    currently running processes."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def put_progress(self, obj):
        pass