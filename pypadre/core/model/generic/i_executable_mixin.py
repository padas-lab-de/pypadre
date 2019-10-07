from abc import ABCMeta, abstractmethod
from typing import Union

from pypadre.core.events.events import signals, CommonSignals, Signaler
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