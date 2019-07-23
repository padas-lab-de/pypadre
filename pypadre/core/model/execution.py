import itertools
import platform
import pypadre.core.visitors.parameter

from collections import OrderedDict
from pypadre.eventhandler import trigger_event, assert_condition
from pypadre.base import MetadataEntity
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.core.validatetraintestsplits import ValidateTrainTestSplits
from pypadre.core.model.sklearnworkflow import SKLearnWorkflow
from pypadre.core.model.run import Run
from pypadre.core.model.split.custom_split import split_obj
from pypadre.core.visitors.mappings import name_mappings, alternate_name_mappings, supported_frameworks
from pypadre.printing.tablefyable import Tablefyable


class Execution(MetadataEntity, Tablefyable):
    """ A execution should save data about the running env and the version of the code on which it was run """

    _id = None
    _metadata = None

    def __init__(self,
                 **options):
        # Validate input types
        self.validate_input_parameters(options=options)
        super().__init__(id_=options.pop("id", None), **options)

        self._runs = []

    def execute(self, parameters=None):
        assert_condition(condition=parameters is None or isinstance(parameters, dict),
                         source=self,
                         message='Incorrect parameter type to the execute function')
        pass

    def validate_input_parameters(self, options):
        pass
