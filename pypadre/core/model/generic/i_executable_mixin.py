from abc import ABCMeta, abstractmethod

from jsonschema import ValidationError

from pypadre.core.events.events import signals, CommonSignals, Signaler
from pypadre.core.validation.validation import ValidateableMixin


@signals(CommonSignals.START, CommonSignals.STOP)
class ExecuteableMixin(Signaler):
    """ This is the mixin for all entities being executable. This allows for signaling a start and a stop of an
    execution. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def send_start(self, **kwargs):
        self.send_signal(CommonSignals.START, self, **kwargs)

    def send_stop(self, **kwargs):
        self.send_signal(CommonSignals.STOP, self, **kwargs)

    def execute(self, *args, **kwargs):
        if not self.is_executable():
            raise ValueError(str(self) + " is not executable.")
        name = self.name if hasattr(self, 'name') else self.__class__.__name__
        self.send_start(message="Execution is starting for {name}".format(name=name))
        execute = self._execute_helper(*args, **kwargs)
        self.send_stop(message="Execution is ending for {name}".format(name=name))
        return execute

    # noinspection PyMethodMayBeStatic
    def is_executable(self, *args, **kwargs):
        return True

    @abstractmethod
    def _execute_helper(self, *args, **kwargs):
        raise NotImplementedError


class ValidateableExecutableMixin(ExecuteableMixin, ValidateableMixin):
    """ This mixin allows for automatically checking if the object is valid before executing."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_executable(self, *args, **kwargs):
        if self.dirty:
            try:
                self.validate()
            except ValidationError as exc:
                self.send_error(str(exc))
        return super().is_executable(*args, **kwargs) and not self.dirty
