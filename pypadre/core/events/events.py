"""
Structure of Event Handling Mechanism in PyPaDRe
Signals can be triggered on Class or Base level.
"""

from blinker import Namespace

from pypadre.core.util.inheritance import SuperStop
from pypadre.core.util.utils import _merge_dict_class_vars


class SignalSchema:
    """
    This class holds a signal name and whether the signal should cascade to the base_signal level.
    """
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
    """
    Common schemas of signals in PaDRE. These are used on the i_model_mixins
    """
    PUT = SignalSchema("put", False)
    DELETE = SignalSchema("delete", False)
    PROGRESS = SignalSchema("progress", False)
    START = SignalSchema("start", False)
    STOP = SignalSchema("stop", False)
    LOG = SignalSchema("log", True)


class PointAccessNamespace(Namespace):
    """
    Namespace extension which allows for dot notation access. This enables the possibility to use decorator connection.
    """
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


# signal name for the signal which is triggered / cascaded to for each other signal
LOG_EVENT = "log_event"

# all classes which where registered via decorator call
signal_classes = set()

# all base signals
base_signals = PointAccessNamespace()
base_signals.signal(LOG_EVENT)


def connect_base_signal(name, fn):
    """
    Connect a fn to a base signal with given name
    :param name:
    :param fn:
    :return:
    """
    if name not in base_signals:
        signal = base_signals.signal(name)
    else:
        signal = base_signals.get(name)
    return signal.connect(fn)


def init_class_signals(clz):
    # TODO cleanup signal initialization
    if clz not in signal_classes:
        # Try to add missing signals
        signals_decorator = signals()
        # noinspection PyTypeChecker
        signals_decorator(clz)


def connect_class_signal(clz, name, fn):
    """
    Connect a a fn to a class signal of given class with given name
    :param clz:
    :param name:
    :param fn:
    :return:
    """
    init_class_signals(clz)
    signal = clz.signals().get(name)
    if signal is None:
        raise ValueError("Signal not defined on class " + str(clz))
    return signal.connect(fn)


def connect(name=None, clz=None):
    """
    Decorator used to decorate methods which are to connect to signals.
    :param name:
    :param clz:
    :return:
    """

    def connect_decorator(fn):
        signal_name = name if name is not None else fn.__name__
        if clz is None:
            connect_base_signal(name, fn)
        else:
            connect_class_signal(clz, name, fn)
        return fn
    return connect_decorator


def signals(*args):
    """
    Decorator used to decorate classes which are to send signals. The decorator takes either string names of signals
    or schema objects
    :param args:
    :return:
    """

    def signals_decorator(cls):
        signal_classes.add(cls)

        if hasattr(cls, "signal_namespace"):
            namespace = PointAccessNamespace(_merge_dict_class_vars(cls, "signal_namespace", Signaler))
        else:
            namespace = PointAccessNamespace()

        for schema in args:
            if isinstance(schema, str):
                schema = SignalSchema(schema)
            signal = namespace.signal(schema.name)

            def make_cascade(name):
                def cascade(sender, **kwargs):
                    base_signals.get(name).send(sender, **kwargs)
                return cascade

            if schema.cascade:
                if schema.name not in base_signals:
                    base_signals.signal(schema.name)

                # make_cascade(signal.name)
                setattr(cls, "_cascade_" + schema.name, make_cascade(signal.name))
                signal.connect(getattr(cls, "_cascade_" + schema.name))
            setattr(cls, "_cascade_" + LOG_EVENT + "_" + schema.name, make_cascade(LOG_EVENT))
            signal.connect(getattr(cls, "_cascade_" + LOG_EVENT + "_" + schema.name))
        cls.signal_namespace = namespace
        return cls
    return signals_decorator


class Signaler(SuperStop):
    """
    Base class of a class being able to send signals.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for enabling signals on init for classes only getting signals by inheritance TODO this might be removed
        """
        signals_decorator = signals()
        # noinspection PyTypeChecker
        signals_decorator(self.__class__)
        super().__init__(*args, **kwargs)

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
            self.signals().get(signal.name).send(*sender, signal=signal, **kwargs)

    @classmethod
    def signals(cls):
        if not hasattr(cls, "signal_namespace"):
            raise ValueError("Namespace not defined on " + str(cls))
        return getattr(cls, "signal_namespace")
