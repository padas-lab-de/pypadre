from pypadre.core.events.events import CommonSignals, connect_subclasses
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
            log_level = kwargs.pop('log_level', "info")

            if log_level == "info":
                for b in self.backends:
                    b.log_info(**kwargs)

            elif log_level == 'warn':
                for b in self.backends:
                    b.log_warn(**kwargs)

            elif log_level == 'error':
                for b in self.backends:
                    b.log_error(**kwargs)

            else:
                for b in self.backends:
                    b.log_warn(**kwargs)

        self.save_signal_fn(log)
