from abc import ABC, abstractmethod, ABCMeta


class ILoggable:
    __metaclass__ = ABCMeta

    @abstractmethod
    def log(self, msg):
        pass
