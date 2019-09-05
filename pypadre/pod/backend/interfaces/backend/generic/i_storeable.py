from abc import abstractmethod, ABCMeta


class IStoreable:
    """ This is the interface for all backends being able to store objects onto some kind of persistence storage."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def get(self, uid):
        pass

    @abstractmethod
    def put(self, obj):
        pass

    @abstractmethod
    def delete_by_id(self, uid):
        pass

    @abstractmethod
    def delete(self, obj):
        pass
