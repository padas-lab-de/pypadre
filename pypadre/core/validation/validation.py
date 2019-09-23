import json
import urllib.request
import warnings
from abc import ABCMeta, abstractmethod
from collections.__init__ import deque
from importlib import resources
from typing import List

from jsonschema import validate, ValidationError, validators
from padre.PaDREOntology import PaDREOntology

from pypadre.core.events import trigger_event


# noinspection PyUnusedLocal,PyShadowingNames,PyProtectedMember
def padre_enum(validator, padre_enum, instance, schema):
    if validator.is_type(instance, "string"):
        if padre_enum is not None:
            if not hasattr(PaDREOntology, padre_enum):
                yield ValidationError("%r is not a valid padre enum" % (padre_enum))
            # TODO cleanup access to enum
            elif instance not in getattr(PaDREOntology, padre_enum)._value2member_map_:
                yield ValidationError("%r is not a valid value entry of padre enum %r" % (instance, padre_enum))


schema_validator = validators.extend(validators.Draft7Validator, validators={"padre_enum": padre_enum}, version="1.0")


class Validateable(object):
    __metaclass__ = ABCMeta
    """ This class implements basic logic for validating the state of it's input parameters """

    # noinspection PyBroadException
    def __init__(self, schema=None, schema_path=None, schema_url=None,
                 schema_resource_package='pypadre.core.resources.schema', schema_resource_name=None, **options):
        # Load schema externally
        if schema is None:
            try:
                if schema_url is not None:
                    with urllib.request.urlopen(schema_url) as url:
                        schema = json.loads(url.read().decode())
            except:
                trigger_event('EVENT_WARN', source=self,
                              message='Failed on loading schema file from url ' + schema_url)

        # Load schema from file
        if schema is None:
            try:
                if schema_path is not None:
                    with open(schema_path, 'r') as f:
                        schema_data = f.read()
                    schema = json.loads(schema_data)
            except:
                trigger_event('EVENT_WARN', source=self,
                              message='Failed on loading schema file from disk ' + schema_path)

        if schema is None:
            try:
                if schema_resource_name is not None and schema_resource_package is not None:
                    with resources.open_text(schema_resource_package, schema_resource_name) as f:
                        schema_data = f.read()
                    schema = json.loads(schema_data)
            except:
                trigger_event('EVENT_WARN', source=self,
                              message='Failed on loading schema file from resources ' + schema_resource_package + '.'
                                      + schema_resource_name)

        self._schema = schema
        # Fail if no schema is provided
        self.validate(**options)

    def validate(self, **kwargs):
        """
        Overwrite this message if you want to add validation logic. Don't forget to call super for jsonschema validation
        :param kwargs:
        :return:
        """
        return self._validate_parameters(kwargs)

    def _validate_parameters(self, options):
        if self._schema is None:
            # TODO make this an error as soon as all validateables are implemented
            warnings.warn("A validateable object needs a schema to validate to: " + str(self), FutureWarning)
            #raise ValueError("A validateable object needs a schema to validate to.")
        else:
            validate(options, self._schema, cls=schema_validator)


class ValidationErrorHandler:
    """ Class to handle errors on the validation of an validatable. """

    def __init__(self, absolute_path=None, validator=None, handle=None):
        self._absolute_path = absolute_path
        self._validator = validator
        self._handle = handle

    @property
    def validator(self):
        return self._validator

    @property
    def absolute_path(self):
        return self._absolute_path

    def handle(self, obj, e, options):
        if (not self._absolute_path or deque(self._absolute_path) == e.absolute_path) and (
                not self._validator or self.validator == e.validator):
            if self._handle is None:
                self._default_handle(e)
            else:
                return self._handle(obj, e, options)
        else:
            raise e

    def _default_handle(self, e):
        print("Validation handler triggered: " + str(self))
        raise e


class ValidateableFactory:

    @staticmethod
    def make(cls, handlers=List[ValidationErrorHandler], **options):
        return ValidateableFactory._make(cls, handlers=handlers, history=[], **options)

    @staticmethod
    def _make(cls, handlers=List[ValidationErrorHandler], history=None, **options):
        try:
            return cls(**options)
        except ValidationError as e:
            # Raise error if we can't handle anything
            if handlers is None:
                raise e
            for handler in handlers:
                new_options = handler.handle(cls, e, options)
                # If the handler could fix the problem return the new value
                if new_options is not None and e not in history and new_options not in history:
                    history.append(e)
                    history.append(new_options)

                    # Try to create object again
                    return ValidateableFactory._make(cls, handlers=handlers, history=history, **new_options)
            # Raise error if we fail
            raise e
