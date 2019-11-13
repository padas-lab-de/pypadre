from abc import ABCMeta, abstractmethod

from pypadre.core.base import MetadataMixin
from pypadre.core.events.events import signals, CommonSignals, Signaler


@signals(CommonSignals.PUT, CommonSignals.DELETE, CommonSignals.GET)
class StoreableMixin(MetadataMixin, Signaler):
    """ This is the interface for all entities being able to signal they are to be persisted, deleted etc."""
    __metaclass__ = ABCMeta

    RETURN_VAL = "return_val"
    # HASH = '__hash'

    @abstractmethod
    def __init__(self, *args, metadata=None, **kwargs):

        in_hash = metadata.get("id", None) if metadata else None

        super().__init__(*args, metadata=metadata, **kwargs)

        cur_hash = self.id

        if in_hash is None:
            by_hash = self.send_get(self, uid=cur_hash)[self.RETURN_VAL]
            if by_hash is not None:
                # TODO check if we get problems here with changing the reference in the constructor. Alternatively we could use a factory.
                self.__class__ = by_hash.__class__
                self.__dict__ = by_hash.__dict__
        elif cur_hash != in_hash:
            self.send_warn("Identity hash " + str(cur_hash) +
                           " of loaded object " + str(self) + " seems to mismatch its stored hash value "
                           + str(in_hash) + ".")

    def send_put(self, **kwargs):
        self.send_signal(CommonSignals.PUT, self, message="Putting object {name}".format(name=self.name), **kwargs)

    def send_delete(self, **kwargs):
        self.send_signal(CommonSignals.DELETE, self, message="Deleting object {name}".format(name=self.name), **kwargs)

    @classmethod
    def send_get(cls, *sender, uid=None, **kwargs):
        callback = {cls.RETURN_VAL: {cls.RETURN_VAL: None}, "uid": uid}
        cls.send_cls_signal(CommonSignals.GET, *sender, **{**callback, **kwargs})
        return callback.get(cls.RETURN_VAL, {cls.RETURN_VAL: None})

    # @property
    # def __hash(self):
    #     if self.HASH in self.metadata:
    #         return self.metadata[self.HASH]
    #     else:
    #         return None
    #

    # @abstractmethod
    # def __eq__(self, other):
    #     raise NotImplementedError()
