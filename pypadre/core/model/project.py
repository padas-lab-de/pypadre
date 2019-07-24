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


class Project(MetadataEntity):
    """ A project should group experiments """

    _id = None
    _metadata = None

    def __init__(self, **options):
        # Validate input types
        self.validate_input_parameters(options=options)

        super().__init__(id_=options.pop("id", None), **options)

        self._experiments = []
        self._sub_projects = []

    def validate_input_parameters(self, options):
        assert_condition(condition=options.get('name', None) is not None, source=self,
                         message="Name cannot be none")
        assert_condition(condition=options.get('description', None) is not None, source=self,
                         message="Description cannot be none")
