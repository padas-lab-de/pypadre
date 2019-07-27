from abc import abstractmethod, ABCMeta


class IValidator:
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def validate(obj):
        """ Validate if given input has feasible settings. This is important for blueprint files and can be used to
        check non-blueprint files """
        pass
