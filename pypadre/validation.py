import json
import re
import urllib.request
from abc import ABCMeta
from collections.__init__ import deque
from typing import List, Sequence

from jsonschema import validate, ValidationError

from pypadre.eventhandler import trigger_event


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

    def handle(self, e):
        if (not self._absolute_path or deque(self._absolute_path) == e.absolute_path) and (
                not self._validator or self.validator == e.validator):
            if self._handle is None:
                self._default_handle(e)
            else:
                return self._handle(self, e)
        else:
            raise e

    def _default_handle(self, e):
        print("Validation handler triggered: " + str(self))
        raise e


class Validateable(object):
    __metaclass__ = ABCMeta
    """ This class implements basic logic for validating the state of it's input parameters """

    # noinspection PyBroadException
    def __init__(self, schema=None, schema_path=None, schema_url=None, **options):
        super(Validateable, self).__init__(**options)
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
                trigger_event('EVENT_WARN', source=self, message='Failed on loading schema file from disk ' + schema_path)

        # Fail if no schema is provided
        if schema is None:
            raise ValueError("A validateable object needs a schema to validate to.")
        validate(options, schema)


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
                value = handler.handle(e)
                # TODO parsing the jsonschema errors is really ugly there has to be a better way (Own Validator?)
                # If the handler could fix the problem return the new value
                if value is not None and e not in history:
                    history.append(e)
                    deq = e.absolute_path
                    new = {}
                    current_level = new
                    while len(deq) > 0:
                        val = deq.popleft()
                        if len(deq) != 0:
                            current_level[val] = {}
                            current_level = current_level[val]

                    if _seq_but_not_str(e.validator_value):
                        for key in e.validator_value:
                            match = re.compile("'(" + re.escape(key) + ")'.*").match(e.message)
                            if match is not None:
                                key = match.group(1)
                                current_level[key] = value
                                break

                    return ValidateableFactory._make(cls, handlers=handlers, history=history, **{**options, **new})
            # Raise error if we fail
            raise e


def _seq_but_not_str(obj):
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))