from abc import ABCMeta, abstractmethod

from pypadre.core.events.events import Signaler, CommonSignals, signals, EVENT_TRIGGERED


@signals(CommonSignals.PUT, CommonSignals.DELETE, CommonSignals.GET)
class StoreableMixin(Signaler):
    """ This is the interface for all entities being able to signal they are to be persisted, deleted etc."""
    __metaclass__ = ABCMeta

    RETURN_VAL = "return_val"

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def send_put(self, **kwargs):
        self.send_signal(CommonSignals.PUT, self, message="Putting object {name}".format(name=self.name), **kwargs)

    def send_delete(self, **kwargs):
        self.send_signal(CommonSignals.DELETE, self, message="Deleting object {name}".format(name=self.name), **kwargs)

    @classmethod
    def send_get(cls, *sender, uid=None, **kwargs):
        callback = {cls.RETURN_VAL: {}, "uid": uid}
        cls.send_cls_signal(CommonSignals.DELETE, *sender, **{**callback, **kwargs})
        return callback.get(cls.RETURN_VAL, {cls.RETURN_VAL: None})


@signals(CommonSignals.PROGRESS)
class ProgressableMixin(Signaler):
    """ This is the interface for all entities being able to signal a progress of their state."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def send_progress(self, *, progress, **kwargs):
        self.send_signal(CommonSignals.PROGRESS, self, progress=progress, **kwargs)


@signals(CommonSignals.LOG)
class LoggableMixin(Signaler):
    """ This is the interface for all entities being able to signal a progress of their state."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    SIGNAL_LOG = CommonSignals.LOG

    class LogLevels:
        LOG = "log"
        INFO = "info"
        WARN = "warn"
        ERROR = "error"

    def send_log(self, *, message, **kwargs):
        self.send_signal(self.SIGNAL_LOG, self, log_level=self.LogLevels.LOG, message=message, **kwargs)

    def send_info(self, *, message, **kwargs):
        self.send_signal(self.SIGNAL_LOG, self, log_level=self.LogLevels.INFO, message=message, **kwargs)

    def send_warn(self, message, condition=True, **kwargs):
        self.send_signal(self.SIGNAL_LOG, log_level=self.LogLevels.WARN, message=message, condition=condition, **kwargs)

    def send_error(self, message, condition=True, **kwargs):
        self.send_signal(self.SIGNAL_LOG, log_level=self.LogLevels.ERROR, message=message, condition=condition, **kwargs)

    def log_event(self, *args, **kwargs):
        self.send_signal(self.SIGNAL_LOG, self, log_level=self.LogLevels.LOG, message=message, **kwargs)

