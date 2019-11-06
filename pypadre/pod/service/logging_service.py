from typing import List, Type

from pypadre.core.model.generic.i_model_mixins import ILoggable
from pypadre.core.util.inheritance import SuperStop
from pypadre.core.validation.validation import ValidationErrorHandler, ValidateableFactory
from pypadre.pod.service.base_service import BaseService
from pypadre.core.events.events import CommonSignals, signals, connect, connect_subclasses


class LoggingService(BaseService):

    def __init__(self, backends, **kwargs):
        b = backends if isinstance(backends, List) else [backends]
        self._backends = [] if backends is None else b

        @connect_subclasses(ILoggable)
        def log(*args, **kwargs):
            """
            Logs a warning to the backend
            :param obj:
            :return:
            """
            log_level = kwargs.pop('log_level')
            message = kwargs.pop('message')

            if log_level == "info":
                for b in self.backends:
                    b.log_warn(message=message, **kwargs)

            elif log_level == 'warn':
                for b in self.backends:
                    b.log_warn(message=message, **kwargs)

            elif log_level == 'error':
                for b in self.backends:
                    b.log_error(message=message, **kwargs)

            else:
                raise ValueError('Undefined Log level given:{log_level}'.format(log_level=log_level))

        self.save_signal_fn(log)

    @property
    def backends(self):
        return self._backends

    @property
    def model_clz(self):
        return self._model_clz

    def create(self, *args, handlers: List[ValidationErrorHandler] = None, **kwargs):
        if handlers is None:
            handlers = []
        return ValidateableFactory.make(self.model_clz, *args, handlers=handlers, **kwargs)

    def list(self, search, offset=0, size=100) -> list:
        """
        Lists all entities matching search.
        :param offset: Offset of the search
        :param size: Size of the search
        :param search: Search object
        :return: Entities
        """
        return super.list(search=search, offset=offset, size=size)

    def put(self, obj, **kwargs):
        """
        Puts the entity
        :param obj: Entity to put
        :return: Entity
        """
        return super.put(obj, kwargs)

    def patch(self, obj):
        """
        Updates the entity
        :param obj: Entity to put
        :return: Entity
        """
        return super.path(obj)

    def get(self, uid):
        """
        Get the entity by id
        :param uid: Id of the entity to get
        :return: Entity
        """
        return super.get(uid)

    def delete(self, obj):
        """
        Delete the entity
        :param obj: Entity to delete
        :return: Entity
        """
        return super.delete(obj)

    @connect_subclasses(ILoggable)
    def log_info(self, **kwargs):
        """
        Logs the information to the backend
        :param obj:
        :return:
        """

        for b in self.backends:
            b.log_info(kwargs)

    @connect_subclasses(ILoggable)
    def log_warn(self, **kwargs):
        """
        Logs a warning to the backend
        :param obj:
        :return:
        """

        for b in self.backends:
            b.log_warn(kwargs)

    @connect_subclasses(ILoggable)
    def log_error(self, **kwargs):
        """
        Logs a warning to the backend
        :param obj:
        :return:
        """

        for b in self.backends:
            b.log_error(kwargs)


