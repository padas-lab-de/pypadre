from abc import ABCMeta, abstractmethod

from pypadre.core.events.events import signals, CommonSignals, Signaler


@signals(CommonSignals.START, CommonSignals.STOP)
class IExecuteable(Signaler):
    """ This is the interface for all entities being able to signal they are to be persisted, deleted etc."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def send_start(self, **kwargs):
        self.send_signal(CommonSignals.START, self, **kwargs)

    def send_stop(self, **kwargs):
        self.send_signal(CommonSignals.STOP, self, **kwargs)

    def execute(self, *args, **kwargs):
        self.send_start()
        execute = self._execute(*args, **kwargs)
        self.send_stop()
        return execute

    @abstractmethod
    def _execute(self, *args, **kwargs):
        raise NotImplementedError
