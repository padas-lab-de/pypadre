from abc import ABCMeta, abstractmethod
from typing import Union

from pypadre.core.events.events import Signaler, CommonSignals, signals
from pypadre.core.model.pipeline.parameters import PipelineParameters


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

    def execute(self, *args, pipeline_parameters: Union[PipelineParameters, dict]=None, parameter_map: PipelineParameters=None, **kwargs):
        if parameter_map is None:
            if pipeline_parameters is None:
                parameter_map = PipelineParameters({})
            if not isinstance(pipeline_parameters, PipelineParameters):
                parameter_map = PipelineParameters(pipeline_parameters)

        self.send_start()
        execute = self._execute(*args, parameter_map=parameter_map, **kwargs)
        self.send_stop()
        return execute

    @abstractmethod
    def _execute(self, *args, **kwargs):
        raise NotImplementedError


@signals(CommonSignals.PUT, CommonSignals.DELETE)
class IStoreable(Signaler):
    """ This is the interface for all entities being able to signal they are to be persisted, deleted etc."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def send_put(self, **kwargs):
        self.send_signal(CommonSignals.PUT, self, **kwargs)

    def send_delete(self, **kwargs):
        self.send_signal(CommonSignals.DELETE, self, **kwargs)


@signals(CommonSignals.PROGRESS)
class IProgressable(Signaler):
    """ This is the interface for all entities being able to signal a progress of their state."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def send_progress(self, *, progress, **kwargs):
        self.send_signal(CommonSignals.PROGRESS, self, progress=progress, **kwargs)


@signals(CommonSignals.LOG)
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
