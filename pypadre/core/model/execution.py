from pypadre.base import MetadataEntity
from pypadre.eventhandler import assert_condition
from pypadre.printing.tablefyable import Tablefyable
from pypadre.util.dict_util import get_dict_attr


class Execution(MetadataEntity, Tablefyable):
    """ A execution should save data about the running env and the version of the code on which it was run """

    @classmethod
    def register_columns(cls):
        cls._register_columns({'hash': get_dict_attr(Execution, 'hash').fget, 'cmd': get_dict_attr(Execution, 'cmd').fget})

    _id = None
    _metadata = None

    def __init__(self, experiment, codehash, command, **options):
        # Validate input types
        self.validate_input_parameters(experiment=experiment, options=options)
        super().__init__(id_=options.pop("id", None), **options)
        self._experiment = experiment
        self._runs = []
        self._hash = codehash
        self._cmd = command

    @property
    def hash(self):
        return self._hash

    @property
    def cmd(self):
        return self._cmd

    @property
    def experiment(self):
        return self._experiment

    def validate_input_parameters(self, experiment, options):
        from pypadre.core.model.experiment import Experiment
        assert_condition(condition=experiment is not None, source=self,
                         message="Experiment cannot be None")
        assert_condition(condition=isinstance(experiment, Experiment), source=self,
                         message="Parameter experiment is not an object of padre.core.Experiment")
