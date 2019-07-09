from abc import abstractmethod, ABCMeta


class IProgressable:
    __metaclass__ = ABCMeta

    @abstractmethod
    def put_progress(self, obj):
        pass