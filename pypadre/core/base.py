from pypadre.util.utils import _Const

####################################################################################################################
#  API Classes
####################################################################################################################
# Constants used for naming conventions. Todo maybe map these in the ontology?
class _Phases(_Const):
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


class _ExperimentEvents(_Const):
    start = "start"
    stop = "stop"


"""
Enum for the different phases of an experiment
"""
exp_events = _ExperimentEvents()