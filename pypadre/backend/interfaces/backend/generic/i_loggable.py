from abc import ABC, abstractmethod, ABCMeta


class ILoggable:
    """ This is the interface for all backends which are able to log interactions into some kind of log store """
    __metaclass__ = ABCMeta

    @abstractmethod
    def log(self, msg):
        pass
