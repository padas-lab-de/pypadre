"""
Structure of Event Handling Mechanism in PyPaDRe
The event emitter fires only a single event called EVENT. In the argument to the event, the actual event name is passed
along with the required arguments to the Event Handler Function.

Each fired event is pushed to a queue, and gets handled one after the other. The logger_list variable contains the list
of loggers that are to be used to handle the events. A single event can be handled by multiple loggers if needed. A
single event can trigger multiple functions too.
"""

from blinker import Namespace

from pypadre.core.util.utils import _merge_dict_class_vars


class SignalSchema:

    def __init__(self, name, cascade=False):
        self._name = name
        self._cascade = cascade

    @property
    def name(self):
        return self._name

    @property
    def cascade(self):
        return self._cascade

    def __hash__(self):
        return str(self.name).__hash__()


class CommonSignals:
    PUT = SignalSchema("put", False)
    DELETE = SignalSchema("delete", False)
    PROGRESS = SignalSchema("progress", False)
    START = SignalSchema("start", False)
    STOP = SignalSchema("stop", False)
    LOG = SignalSchema("log", True)


class PointAccessNamespace(Namespace):
    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super().__delitem__(key)
        del self.__dict__[key]


base_signals = PointAccessNamespace()


def signals(*args):

    def signals_decorator(cls):

        if hasattr(cls, "signal_namespace"):
            namespace = PointAccessNamespace(_merge_dict_class_vars(cls, "signal_namespace", Signaler))
        else:
            namespace = PointAccessNamespace()

        for schema in args:
            if isinstance(schema, str):
                schema = SignalSchema(schema)
            signal = namespace.signal(schema.name)
            if schema.cascade:
                if schema.name not in base_signals:
                    base_signals.signal(schema.name)

                def make_cascade(name):
                    def cascade(sender, **kwargs):
                        base_signals.get(name).send(sender, **kwargs)

                    return cascade

                # make_cascade(signal.name)
                setattr(cls, "_cascade_" + schema.name, make_cascade(signal.name))

                signal.connect(getattr(cls, "_cascade_" + schema.name))
        cls.signal_namespace = namespace
        return cls
    return signals_decorator


class Signaler:

    def __init__(self, *args, **kwargs):
        """
        Constructor for enabling signals on init for classes only getting signals by inheritance
        """
        signals_decorator = signals()
        # noinspection PyTypeChecker
        signals_decorator(self.__class__)

    def send_signal(self, signal: SignalSchema, condition=None, *sender, **kwargs):
        if condition is None or condition:
            if len(sender) == 0:
                sender = [self]
            if signal.name not in self.signals():
                # Try to add missing signals
                signals_decorator = signals()
                # noinspection PyTypeChecker
                signals_decorator(self.__class__)
                if signal.name not in self.signals():
                    raise ValueError("Signal is not existing on " + str(self.__class__))
            self.signals().get(signal.name).send(*sender, **kwargs)

    @classmethod
    def signals(cls):
        if not hasattr(cls, "signal_namespace"):
            raise ValueError("Namespace not defined on " + str(cls))
        return getattr(cls, "signal_namespace")
