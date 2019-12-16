from pypadre.core.events.events import CommonSignals, connect_subclasses
from pypadre.core.model.generic.i_executable_mixin import ExecuteableMixin
from pypadre.core.model.generic.i_model_mixins import LoggableMixin
from pypadre.pod.service.base_service import ServiceMixin


class LoggingService(ServiceMixin):

    def __init__(self, backends, *args, **kwargs):
        super().__init__(backends, *args, **kwargs)

        @connect_subclasses(LoggableMixin, name=CommonSignals.LOG.name)
        def log(*args, **kwargs):
            """
            Logs a warning to the backend
            :param obj:
            :return:
            """
            log_level = kwargs.pop('log_level', LoggableMixin.LogLevels.INFO)

            if log_level == LoggableMixin.LogLevels.INFO:
                for b in self.backends:
                    b.log_info(**kwargs)

            elif log_level == LoggableMixin.LogLevels.WARN:
                for b in self.backends:
                    b.log_warn(**kwargs)

            elif log_level == LoggableMixin.LogLevels.ERROR:
                for b in self.backends:
                    b.log_error(**kwargs)

            else:
                for b in self.backends:
                    b.log_warn(message="Incorrect logging level specified: {log_level}".format(log_level=log_level))
                    b.log_warn(**kwargs)

                #  raise ValueError("Incorrect logging level specified: {log_level}".format(log_level=log_level))
        self.save_signal_fn(log)

        @connect_subclasses(ExecuteableMixin, name=CommonSignals.START.name)
        def log_event(sender, *args, **kwargs):

            if isinstance(sender, LoggableMixin):
                for b in self.backends:
                    b.log_info(**kwargs)

        self.save_signal_fn(log_event)

