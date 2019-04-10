from padre.utils import _const
####################################################################################################################
#  API Classes
####################################################################################################################
class _Phases(_const):
    experiment = "experiment"
    run = "run"
    split = "split"
    fitting = "fitting/training"
    validating = "validating"
    inferencing = "inferencing/testing"


"""
Enum for the different phases of an experiment
"""
phases = _Phases()


class _ExperimentEvents(_const):
    start = "start"
    stop = "stop"


"""
Enum for the different phases of an experiment
"""
exp_events = _ExperimentEvents()