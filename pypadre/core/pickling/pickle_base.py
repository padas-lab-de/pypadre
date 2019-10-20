from abc import abstractmethod, ABCMeta
from functools import wraps

from pypadre.core.util.inheritance import SuperStop


class Pickleable(SuperStop):
    """ Class allows for setting fields to transient and exposes which fields are set to
    transient with a simple method. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def transient_fields(self):
        raise NotImplementedError()

    def __getstate__(self):
        state = dict(self.__dict__)
        for field in self.transient_fields():
            del state[field]
        return state


TRANSIENT_FIELDS = "transient_fields"


def transient(*args: str):
    """ deprecated! marks given field names as transient for pickle """

    def transient_decorator(clz):
        setattr(clz, TRANSIENT_FIELDS, args)

        @wraps(clz)
        def get_state(self):
            # TODO super call?
            state = dict(self.__dict__)
            for field in getattr(clz, TRANSIENT_FIELDS):
                del state[field]
            return state
        setattr(clz, "__getstate__", get_state)
        return get_state
    return transient_decorator
