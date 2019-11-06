from abc import ABCMeta, abstractmethod

from pypadre.core.events.events import Signaler, CommonSignals, signals


@signals(CommonSignals.PUT, CommonSignals.DELETE, CommonSignals.GET)
class IStoreable(Signaler):
    """ This is the interface for all entities being able to signal they are to be persisted, deleted etc."""
    __metaclass__ = ABCMeta

    RETURN_VAL = "return_val"

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def send_put(self, **kwargs):
        self.send_signal(CommonSignals.PUT, self, **kwargs)

    def send_delete(self, **kwargs):
        self.send_signal(CommonSignals.DELETE, self, **kwargs)

    @classmethod
    def send_get(cls, *sender, uid=None, **kwargs):
        callback = {cls.RETURN_VAL: {}, "uid": uid}
        cls.send_cls_signal(CommonSignals.DELETE, *sender, **{**callback, **kwargs})
        return callback.get(cls.RETURN_VAL, {cls.RETURN_VAL: None})


@signals(CommonSignals.PROGRESS)
class IProgressable(Signaler):
    """ This is the interface for all entities being able to signal a progress of their state."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def send_progress(self, *, progress, **kwargs):
        self.send_signal(CommonSignals.PROGRESS, self, progress=progress, **kwargs)


@signals(CommonSignals.LOG, CommonSignals.LOG_INFO, CommonSignals.LOG_WARN, CommonSignals.LOG_ERROR)
class ILoggable(Signaler):
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
        self.send_signal(CommonSignals.LOG, self, log_level=self.LogLevels.LOG, message=message, **kwargs)

    def send_info(self, *, message, **kwargs):
        self.send_signal(CommonSignals.LOG, self, log_level=self.LogLevels.INFO, message=message, **kwargs)

    def send_warn(self, message, condition=None, **kwargs):
        self.send_signal(CommonSignals.LOG, log_level=self.LogLevels.WARN, message=message, condition=condition, **kwargs)

    def send_error(self, message, condition=None, **kwargs):
        self.send_signal(CommonSignals.LOG, log_level=self.LogLevels.ERROR, message=message, condition=condition, **kwargs)
