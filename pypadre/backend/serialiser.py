import pickle
import json
import msgpack_numpy as mn

# TODO : we should support faster / more sophisticated / cross-plattform serialisation. for example using pyarrow
# TODO: write PickleSerialiser Test
from pypadre.backend.interfaces.i_serializer import Serializer


class PickleSerializer(Serializer):
    """
    Serialiser using pythons pickle.
    """

    @staticmethod
    def serialise(obj):
        """
        serializes the object and returns a byte object
        :param obj: object to serialise
        :return: byte object (TODO: Specify more precise)
        """
        return pickle.dumps(obj)

    @staticmethod
    def deserialize(buffer):
        """
        Deserialize a object
        :param buffer:
        :return:
        """
        return pickle.loads(buffer)


class JSonSerializer(Serializer):

    @staticmethod
    def serialise(obj):
        return json.dumps(obj)

    @staticmethod
    def deserialize(buffer):
        return json.loads(buffer)


class MsgPack(Serializer):

    @staticmethod
    def serialise(obj):
        return mn.dumps(obj)

    @staticmethod
    def deserialize(buffer):
        return mn.loads(buffer)
