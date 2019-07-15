from abc import ABC, abstractmethod, ABCMeta


class IStoreable:
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
