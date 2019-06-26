from pypadre.base import LoggerBase, phases, exp_events
from datetime import datetime
import sys


class ErrorLogger(LoggerBase):
    """
    Logs only the warning and error messages to the standard output.
    This class informs the user of warnings and error messages even if the user does not provide a logger
    """

    def warn(self, condition, source, message):
        if not condition:
            sys.stderr.write(str(source) + ":\t" + message + "\n")

    def error(self, source, message):
        # Write the error to the standard error stream
        sys.stderr.write(str(source) + ":\t" + message + "\n")


class PadreLogger(LoggerBase):
    """
    Class for logging output, warnings and errors
    """
    _file = None
    _backend = None

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

    def log_start_experiment(self, experiment, append_runs: bool =False):
        """
        This function handles the start of an experiment.

        :param experiment: Experiment object to be logged
        :param append_runs: Whether to append the runs to the experiment if it exists or remove the directory

        :return:
        """
        if self._backend is not None:
            # Todo a bug happens here
            self._backend.put_experiment(experiment, append_runs=append_runs)
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
