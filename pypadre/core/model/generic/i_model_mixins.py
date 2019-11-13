from abc import ABCMeta, abstractmethod

from pypadre.core.events.events import Signaler, CommonSignals, signals


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

