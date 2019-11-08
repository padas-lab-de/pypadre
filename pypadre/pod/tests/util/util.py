import sys

from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.core.events.events import connect_base_signal, EVENT_TRIGGERED, CommonSignals
from pypadre.core.model.generic.i_model_mixins import LoggableMixin


def create_sklearn_test_pipeline(*, estimators, **kwargs):
    def sklearn_pipeline():
        from sklearn.pipeline import Pipeline
        return Pipeline(estimators)

    return SKLearnPipeline(pipeline_fn=sklearn_pipeline, **kwargs)


def _log(sender, *, message, log_level="", **kwargs):
    if log_level is "":
        print(str(sender) + ": " + message)
    else:
        if log_level is LoggableMixin.LogLevels.ERROR:
            sys.stderr.write(log_level.upper() + ": " + str(sender) + ": " + message)
        else:
            sys.stdout.write(log_level.upper() + ": " + str(sender) + ": " + message)


def _log_event(sender, *, signal, **kwargs):
    _log(sender, message="Triggered " + str(signal.name) + " with " + str(kwargs))


def connect_log_to_stdout():
    connect_base_signal(CommonSignals.LOG.name, _log)


def connect_event_to_stdout():
    connect_base_signal(EVENT_TRIGGERED, _log_event)
