"""
Modul containing basic padre datastructures
"""
from datetime import datetime
from time import time

from pypadre.core.events import LoggerBase
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


class PadreLogger(LoggerBase):
    """
    Class for logging output, warnings and errors
    """
    _file = None
    _backend = None
    _app = None

    def __init__(self, app):
        self._app = app

    def warn(self, condition, source, message):
        if not condition:
            # sys.stderr.write(str(source) + ":\t" + message + "\n")
            if self.has_backend():
                self.backend.log("WARN: " + str(datetime.now())[:-3] + " " + str(source) + ":\t" + message + "\n")

    def error(self, source, message):

        if self.has_backend():
            self.backend.log("ERROR: " + str(datetime.now())[:-3] + " " + str(source) + ":\t" + message + "\n")

    def log(self, source, message, padding=""):

        if self.has_backend():
            self.backend.log("INFO: " + str(datetime.now())[:-3] + " " + padding + str(source) + ":\t" + message + "\n")

    def log_start_experiment(self, experiment, append_runs: bool = False):
        """
        This function handles the start of an experiment.

        :param experiment: Experiment object to be logged
        :param append_runs: Whether to append the runs to the experiment if it exists or remove the directory

        :return:
        """
        if self._app is not None:
            # Todo a bug happens here
            self._app.experiments.put(experiment)
            self.log_event(experiment, exp_events.start, phase=phases.experiment)

    def log_stop_experiment(self, experiment):
        """
        This function stops the logging of an experiment, handles the closing of the backends

        param experiment: Experiment that is being logged

        :return:
        """
        if self._backend is not None:
            self.log_event(experiment, exp_events.stop, phase=phases.experiment)
            self._backend.log_end_experiment()

    def log_start_execution(self, execution, append_runs: bool = False):
        pass

    def log_stop_execution(self, execution):
        pass

    def log_start_run(self, run):
        """
        This function handles the start of a run

        :param run: The run object to be logged

        :return:
        """
        if self.has_backend():
            self._backend.put_run(run.experiment, run)
            self.log_event(run, exp_events.start, phase=phases.run)

    def log_stop_run(self, run):
        """
        This function handles the end of a run

        :param run: The run object

        :return:
        """
        if self._backend is not None:
            self.log_event(run, exp_events.stop, phase=phases.run)

    def log_start_split(self, split):
        """
        This function handles the start of a split

        :param split: The split object

        :return:
        """
        if self.has_backend():
            self._backend.put_split(split.run.experiment, split.run, split)
            self.log_event(split, exp_events.start, phase=phases.split)

    def log_stop_split(self, split):
        """
        This function handles the logging at the end of a split

        :param split: The split object

        :return:
        """
        if self.has_backend():
            self._backend.put_results(split.run.experiment, split.run, split, split.run.workflow.results)
            self._backend.put_metrics(split.run.experiment, split.run, split, split.run.workflow.metrics)
            self.log_event(split, exp_events.stop, phase=phases.split)

    def log_event(self, source, kind=None, **parameters):
        """
        Logs an event to the backend

        :param source: Function calling the log event
        :param kind: Start/Stop of event
        :param parameters: Parameters to be written to the log
        :return:
        """
        if self._backend is not None:
            self.log(source, "%s: %s" % (str(kind), "\t".join([str(k) + "=" + str(v)
                                                               for k, v in parameters.items()])),
                     self._padding(source))

    def log_score(self, source, **parameters):
        # todo signature not yet fixed. might change. unclear as of now
        if self._backend is not None:
            self.log(source, "%s" % ("\t".join([str(k) + "=" + str(v) for k, v in parameters.items()])),
                     "\t\t")

    def log_result(self, source, **parameters):
        # todo signature not yet fixed. might change. unclear as of now
        if self._backend:
            self.backend.log("RESULT: " + str(datetime.now())[:-3] + " " +
                             "\t\t" + str(source) + ":\t" + "%s" % ("\t".join([str(k) + "=" + str(v)
                                                                               for k, v in parameters.items()])) + "\n")

    def put_experiment_configuration(self, experiment):
        """
        Writes the experiment configuration to the backend
        :param experiment:
        :return:
        """
        if self._backend:
            self._backend.put_experiment_configuration(experiment=experiment)

    def log_experiment_progress(self, curr_value, limit, phase):
        """
        Reports the progress of the experiment to the backend
        :param curr_value: Current Experiment that is executing
        :param limit: Total experiments
        :param phase: Start/Stop of the experiment
        :return:
        """
        if self._backend:
            self._backend.log_experiment_progress(curr_value=curr_value, limit=limit, phase=phase)

    def log_run_progress(self, curr_value, limit, phase):
        """
        Reports the overall progress of the run to the backend
        :param curr_value: Current run that is executing
        :param limit: Total number of runs
        :param phase: Start/stop of the run
        :return:
        """
        if self._backend:
            self._backend.log_run_progress(curr_value=curr_value, limit=limit, phase=phase)

    def log_split_progress(self, curr_value, limit, phase):
        """
        Reports the overall progress of the split to the backend
        :param curr_value: Current Experiment that is executing
        :param limit: Total number of splits
        :param phase:
        :return:
        """
        if self._backend:
            self._backend.log_split_progress(curr_value=curr_value, limit=limit, phase=phase)

    def log_progress(self, message, curr_value, limit, phase):
        """
        Reports the progress of a function to the backend
        :param message: Message to be written to the backend
        :param curr_value: Current execution value
        :param limit: Total number
        :param phase: start/Stop
        :return:
        """
        if self._backend:
            self._backend.log_progress(message=message, curr_value=curr_value, limit=limit, phase=phase)

    def log_model(self, model, framework, filename, finalmodel=False):
        """
        Logs an intermediate model to the backend
        :param model: Model to be logged
        :param filename: Name of the intermediate model
        :param framework: Framework of the model
        :param finalmodel: Boolean value indicating whether the model is the final one or not
        :return:
        """
        if self._backend:
            self._backend.log_model(model=model, framework=framework,
                                    filename=filename, finalmodel=finalmodel)

    def _padding(self, source):

        if source.__str__().lower().find('split') > -1:
            return "\t\t"

        elif source.__str__().lower().find('run') > -1:
            return "\t"

        else:
            return ""

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        self._backend = backend

    def has_backend(self):
        return self._backend is not None


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
