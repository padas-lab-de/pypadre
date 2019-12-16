# Copyright 2012 Brian Waldon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-validating model for arbitrary objects"""

import copy
import warnings

import jsonpatch
import jsonschema
import six

from . import exceptions


class Model(dict):
    def __init__(self, *args, **kwargs):
        # we overload setattr so set this manually
        d = dict(*args, **kwargs)

        dict.__init__(self, d)
        self._dirty = True
        self.__dict__["changes"] = {}
        self.__dict__["__original__"] = copy.deepcopy(d)

    def __setitem__(self, key, value):
        mutation = dict(self.items())
        mutation[key] = value
        self._dirty = True
        dict.__setitem__(self, key, value)

        self.__dict__["changes"][key] = value

    def __delitem__(self, key):
        mutation = dict(self.items())
        del mutation[key]
        self._dirty = True
        dict.__delitem__(self, key)

    def __getattr__(self, key):
        if key is not "_dirty":
            try:
                return self.__getitem__(key)
            except KeyError:
                raise AttributeError(key)
        else:
            super().__getattribute__(key)

    def __setattr__(self, key, value):
        if key is not "_dirty":
            self.__setitem__(key, value)
        else:
            super().__setattr__(key, value)

    def __delattr__(self, key):
        self.__delitem__(key)

    # BEGIN dict compatibility methods

    def clear(self):
        raise exceptions.InvalidOperation()

    def pop(self, key, default=None):
        raise exceptions.InvalidOperation()

    def popitem(self):
        raise exceptions.InvalidOperation()

    def copy(self):
        return copy.deepcopy(dict(self))

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return copy.deepcopy(dict(self), memo)

    def update(self, other):
        mutation = dict(self.items())
        mutation.update(other)
        self._dirty = True
        dict.update(self, other)

    def iteritems(self):
        return six.iteritems(copy.deepcopy(dict(self)))

    def items(self):
        return copy.deepcopy(dict(self)).items()

    def itervalues(self):
        return six.itervalues(copy.deepcopy(dict(self)))

    def values(self):
        return copy.deepcopy(dict(self)).values()

    # END dict compatibility methods

    @property
    def patch(self):
        """Return a jsonpatch object representing the delta"""
        original = self.__dict__["__original__"]
        return jsonpatch.make_patch(original, dict(self)).to_string()

    @property
    def changes(self):
        """Dumber version of 'patch' method"""
        deprecation_msg = "Model.changes will be removed in warlock v2"
        warnings.warn(deprecation_msg, DeprecationWarning, stacklevel=2)
        return copy.deepcopy(self.__dict__["changes"])

    @property
    def dirty(self):
        return self._dirty

    def validate(self):
        """Apply a JSON schema to an object"""
        if self.resolver is not None:
            jsonschema.validate(self, self.schema, cls=self.validator, resolver=self.resolver)
            self._dirty = False
        else:
            jsonschema.validate(self, self.schema, cls=self.validator)
            self._dirty = False
