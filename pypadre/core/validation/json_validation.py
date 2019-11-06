import json
import urllib.request
import warnings
from abc import ABCMeta
from importlib import resources

from pypadre.core.validation.json_schema import padre_schema_validator
from pypadre.core.validation.validation import ValidateableMixin
from pypadre.core.validation.warlock.core import model_factory


def make_model(schema=None, schema_path=None, schema_url=None,
               schema_resource_package='pypadre.core.resources.schema', schema_resource_name=None,
               schema_validator=padre_schema_validator):
    """ This function creates a validation model from a jsonschema file. This model can be instantiated and performe
    self validation. """
    if schema is None:
        try:
            if schema_url is not None:
                with urllib.request.urlopen(schema_url) as url:
                    schema = json.loads(url.read().decode())
        except:
            warnings.warn(message='Failed on loading schema file from url ' + schema_url)

    # Load schema from file
    if schema is None:
        try:
            if schema_path is not None:
                with open(schema_path, 'r') as f:
                    schema_data = f.read()
                schema = json.loads(schema_data)
        except:
            warnings.warn(message='Failed on loading schema file from disk ' + schema_path)

    if schema is None:
        try:
            if schema_resource_name is not None and schema_resource_package is not None:
                with resources.open_text(schema_resource_package, schema_resource_name) as f:
                    schema_data = f.read()
                schema = json.loads(schema_data)
        except:
            warnings.warn(message='Failed on loading schema file from resources ' + schema_resource_package + '.'
                                  + schema_resource_name)
    if schema is None:
        return None
    return model_factory(schema, schema_validator)


class ModelHolderMixin(ValidateableMixin):
    __metaclass__ = ABCMeta
    """ This class implements basic logic for validating the state of it's input parameters """

    # noinspection PyBroadException
    def __init__(self, *args, model_clz=None, metadata, **kwargs):
        if metadata is None:
            metadata = {}

        self._val_model = metadata
        self._model_clz = model_clz

        super().__init__(*args, metadata=metadata, **kwargs)

    def validate(self, **kwargs):
        """
        Overwrite this message if you want to add validation logic. Don't forget to call super for jsonschema validation
        :param kwargs:
        :return:
        """
        self._validate_metadata(kwargs.pop("metadata", {}))

    def _validate_metadata(self, metadata):
        if self._model_clz is None:
            # TODO make this an error as soon as all validateables are implemented
            self.send_warn(message="A validateable object needs a model class to validate to: " + str(self))
            # warnings.warn("A validateable object needs a schema to validate to: " + str(self), FutureWarning)
            # raise ValueError("A validateable object needs a schema to validate to.")
        else:
            val_model = self._model_clz(**metadata)
            val_model.validate()
            self._val_model = val_model

    @property
    def dirty(self):
        return getattr(self._val_model, "dirty", False)
