from pypadre.core.events.events import LOG_EVENT, connect_base_signal
from pypadre.pod.service.base_service import ServiceMixin


class LoggingService(ServiceMixin):

    def __init__(self, backends, *args, **kwargs):
        super().__init__(backends, *args, **kwargs)

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
                # raise ValueError('Undefined Log level given:{log_level}'.format(log_level=log_level))

        connect_base_signal(LOG_EVENT, log)
        self.save_signal_fn(log)
