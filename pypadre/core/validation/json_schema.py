import re
from typing import Sequence

from pypadre.core.validation.validation import ValidationErrorHandler


class JsonSchemaRequiredHandler(ValidationErrorHandler):
    """
    Handle a required error of jsonschema validation.
    """

    def __init__(self, absolute_path=None, validator=None, get_value=None):
        super().__init__(absolute_path, validator, get_value)

    @property
    def validator(self):
        return self._validator

    @property
    def absolute_path(self):
        return self._absolute_path

    def handle(self, obj, e, options):
        return self.update_options(e, options, super().handle(obj, e, options))

    @staticmethod
    def update_options(e, options, value):
        deq = e.absolute_path
        new = {}
        current_level = new

        # Build path for the options dict
        while len(deq) > 0:
            val = deq.popleft()
            if len(deq) != 0:
                current_level[val] = {}
                current_level = current_level[val]

        # Insert value in correct field in options dict
        if _seq_but_not_str(e.validator_value):
            for key in e.validator_value:

                # TODO parsing the jsonschema errors is really ugly there has to be a better way (Own Validator?)
                match = re.compile("'(" + re.escape(key) + ")'.*").match(e.message)
                if match is not None:
                    key = match.group(1)
                    current_level[key] = value
                    break
        return {**options, **new}


def _seq_but_not_str(obj):
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))
