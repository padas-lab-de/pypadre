from abc import abstractmethod, ABCMeta


class Serializer:
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def serialise(obj):
        """
        serializes the object and returns a byte object
        :param obj: object to serialise
        :return: byte object (TODO: Specify more precise)
        """
        pass

    @staticmethod
    @abstractmethod
    def deserialize(buffer):
        """
        Deserialize a object
        :param buffer:
        :return:
        """
        pass
