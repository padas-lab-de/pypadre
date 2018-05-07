# This file contains the TimeKeeper class.
# It is implemented as a proof of concept for PyPaDRe
import time

# All the default values for TimeKeeper class and ContentsofTimer class
DEFAULT_PRIORITY = 3
DEFAULT_TIMER_DESCRIPTION = "default_timer"
DEFAULT_TIMER_NAME = "default_timer"
DEFAULT_TIMER_VALUE = 0

class TimeKeeper:
    # This class creates a dictionary of timers.
    # It has a log timer function that would log each timer passed to it.
    # If the timer name is already present in the dictionary, the timer is
    # popped out and the execution time is calculated.

    # Priorities of the timers are defined
    NO_LOGGING = 0
    HIGH_PRIORITY = 1
    MED_PRIORITY = 2
    LOW_PRIORITY = 3

    def __init__(self, priority):
        self._timers = dict()
        self._priority = priority

    def __del__(self):
        if len(self._timers) > 0:
            print("Error: The following Timers still present in the list")
            for key in self._timers:
                print(key)


    def log_timer(self, timer_name=DEFAULT_TIMER_NAME,
                  priority=DEFAULT_PRIORITY,
                  description=DEFAULT_TIMER_DESCRIPTION):
        # If there is no timer by the name, add the timer to the dictionary
        # Else pop out the timer, and check whether the priority of the timer is
        # equal to or higher than the priority of the program.
        # If it is, then print the description and timer
        if self._timers.get(timer_name) is None:
            new_timer = TimerContents(priority=priority,
                                        description=description,
                                        time=time.time())
            self._timers[timer_name]=new_timer

        else:
            old_timer = self._timers.pop(timer_name,None)
            if(old_timer.get_timer_priority() <= self._priority):
                print(old_timer.get_description(),time.time()-old_timer.get_time(), sep=": ")


class TimerContents:
    # This class contains the contents to be displayed and calculated,
    # using the TimeKeeper Class.
    # The class stores three values and there are three get attributes
    # corresponding to each value. There is no set attribute and the values are
    # initialized during object creation itself.

    def __init__(self, priority = DEFAULT_PRIORITY,
                 description = DEFAULT_TIMER_DESCRIPTION,
                 time = DEFAULT_TIMER_VALUE):
        self._timer_desc = description
        self._timer_priority = priority
        self._time = time

    def get_description(self):
        return self._timer_desc

    def get_time(self):
        return self._time

    def get_timer_priority(self):
        return self._timer_priority