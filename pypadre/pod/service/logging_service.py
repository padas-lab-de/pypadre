from pypadre.core.events.events import CommonSignals, connect_subclasses, EVENT_TRIGGERED
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
                    b.log_warn(**kwargs)

                #  raise ValueError("Incorrect logging level specified: {log_level}".format(log_level=log_level))
        self.save_signal_fn(log)

        # @connect_subclasses(LoggableMixin, name=EVENT_TRIGGERED)
        # def log_event(*args, **kwargs):
        #     for b in self.backends:
        #         b.log_info(**kwargs)

        # self.save_signal_fn(log_event)

