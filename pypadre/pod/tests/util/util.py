import sys

from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.core.events.events import connect_base_signal, LOG_EVENT
from pypadre.core.model.generic.i_model_mixins import ILoggable


def create_sklearn_test_pipeline(*, estimators, **kwargs):
    def sklearn_pipeline():
        from sklearn.pipeline import Pipeline
        return Pipeline(estimators)

    return SKLearnPipeline(pipeline_fn=sklearn_pipeline, **kwargs)


def _log(sender, *, message, log_level="", **kwargs):
    if log_level is "":
        print(str(sender) + ": " + message)
    else:
        if log_level is ILoggable.LogLevels.ERROR:
            sys.stderr.write(log_level.upper() + ": " + str(sender) + ": " + message)
        else:
            sys.stdout.write(log_level.upper() + ": " + str(sender) + ": " + message)


def _log_event(sender, *, signal, **kwargs):
    _log(sender, message="Triggered " + str(signal.name) + " with " + str(kwargs))


def connect_log_to_stdout():
    connect_base_signal("log", _log)


def connect_event_to_stdout():
    connect_base_signal(LOG_EVENT, _log_event)
