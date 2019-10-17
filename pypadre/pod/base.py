"""
Modul containing basic padre datastructures
"""
from datetime import datetime
from time import time

from pypadre.core.util.utils import _Const

"""
Enum for the different phases of an experiment
"""


class _Phases(_Const):
    experiment = "experiment"
    preprocessing = "preprocessing"
    run = "run"
    split = "split"
    fitting = "fitting/training"
    validating = "validating"
    inferencing = "inferencing/testing"


"""
Enum for the different phases of an experiment
"""
phases = _Phases()


class _ExperimentEvents(_Const):
    start = "start"
    stop = "stop"


"""
Enum for the different phases of an experiment
"""
exp_events = _ExperimentEvents()

class _timer_priorities(_Const):
    """
    Constant class detailing the different priorites possible for a timer.
    """
    NO_LOGGING = 0
    HIGH_PRIORITY = 1
    MED_PRIORITY = 2
    LOW_PRIORITY = 3


class _timer_defaults(_Const):
    """
    Constant class detailing the different default values for a timer.
    """
    DEFAULT_PRIORITY = 3
    DEFAULT_TIMER_DESCRIPTION = None
    DEFAULT_TIMER_NAME = 'default_timer'
    DEFAULT_TIME = 0.0


timer_priorities = _timer_priorities
timer_defaults = _timer_defaults


class TimeKeeper:
    """
    This class creates a dictionary of timers.
    It has a log timer function that would log each timer passed to it.
    If the timer name is already present in the dictionary, the timer is
    popped out and the execution time is calculated. Multiple timers can be kept
    track of this way.
    Priorities of the timers are also defined
    The timer is logged only if the priority of the timer is equal to or higher than the
    priority defined while initializing.
    Priorities Available.
    NO_LOGGING: None of the timers are logged.
    HIGH_PRIORITY: Only timers having high priority are logged.
    MED_PRIORITY: Only timers with medium or higher priority are logged.
    LOW_PRIORITY: All timers are logged.
    """

    def __init__(self, priority):
        """
        This initializes the TimeKeeper class.
        :param priority: The current logging priority
        """
        self._timers = dict()
        self._priority = priority

    def __del__(self):
        """
        This is the destructor function of the TimeKeeper class.
        The purpose of the destructor is to check whether any timers are left in the TimeKeeper class at
        the end of execution.
        :return: None
        """
        if len(self._timers) > 0:
            print("Error: The following Timers still present in the list")
            for key in self._timers:
                print(key)

    def log_timer(self, timer_name=timer_defaults.DEFAULT_TIMER_NAME,
                  priority=timer_defaults.DEFAULT_PRIORITY,
                  description=timer_defaults.DEFAULT_TIMER_DESCRIPTION):
        """

        :param timer_name: Name of the unique timer. If timer is already present,
        the timer would be popped out and duration recorded.
        :param priority: priority of the timer.
        :param description: The string description of what the timer measures.
        :return: Description of the the timer and its duration.
        """

        # If there is no timer by the name, add the timer to the dictionary
        # Else pop out the timer, and check whether the priority of the timer is
        # equal to or higher than the priority of the program.
        # If it is, then print the description and timer
        if self._timers.get(timer_name) is None:
            new_timer = TimerContents(priority=priority,
                                      description=description,
                                      curr_time=time())
            self._timers[timer_name] = new_timer

        else:
            old_timer = self._timers.pop(timer_name, None)
            if old_timer.get_timer_priority() <= self._priority:
                return old_timer.get_description(), time() - old_timer.get_time()

    def start_timer(self, timer_name=timer_defaults.DEFAULT_TIMER_NAME,
                    priority=timer_defaults.DEFAULT_PRIORITY,
                    description=timer_defaults.DEFAULT_TIMER_DESCRIPTION):
        """
        Starts a unique timer with the key as timer_name
        :param timer_name: Unique name of the timer
        :param priority: Priority of the timer
        :param description: Describes the purpose of the timer
        :return: None
        """
        if self._timers.get(timer_name) is None:
            new_timer = TimerContents(priority=priority,
                                      description=description,
                                      curr_time=time())
            self._timers[timer_name] = new_timer

    def stop_timer(self, timer_name):
        """
        Stops a timer and measures its duration
        :param timer_name:
        :return: Description of the timer and its duration
                 None if a timer with its unique name is not present
        """
        if timer_name is None:
            return None

        old_timer = self._timers.pop(timer_name, None)
        if old_timer.priority <= self._priority:
            return old_timer.description, time() - old_timer.time


class _timer_defaults(_Const):
    """
    Constant class detailing the different default values for a timer.
    """
    DEFAULT_PRIORITY = 3
    DEFAULT_TIMER_DESCRIPTION = None
    DEFAULT_TIMER_NAME = 'default_timer'
    DEFAULT_TIME = 0.0


class TimerContents:
    """
    This class contains the contents to be displayed and calculated,
    using the TimeKeeper Class.
    The class stores three values and there are three get attributes
    corresponding to each value. There is no set attribute and the values are
    initialized during object creation itself.
    """

    def __init__(self, priority=timer_defaults.DEFAULT_PRIORITY,
                 description=timer_defaults.DEFAULT_TIMER_DESCRIPTION,
                 curr_time=timer_defaults.DEFAULT_TIME):
        """
        The initialization of the Timer class. All the arguments are given default values
        which are present in timer_defaults.
        :param priority: The priority of the timer.
        :param description: The description of the timer.
        :param curr_time: The time to start logging.
        """
        self._timer_desc = description
        self._timer_priority = priority
        self._time = curr_time

    @property
    def description(self):
        return self._timer_desc

    @property
    def time(self):
        return self._time

    @property
    def priority(self):
        return self._timer_priority


# TODO: A better way of using the default timer
# A static object shared throughout the instances of _LoggerMixin
default_timer = TimeKeeper(timer_defaults.DEFAULT_PRIORITY)
