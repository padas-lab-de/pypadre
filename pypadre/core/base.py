import uuid
from abc import ABCMeta, abstractmethod

from pypadre.core.printing.tablefyable import Tablefyable
from pypadre.core.util.inheritance import SuperStop
from pypadre.core.util.utils import _Const
from pypadre.core.validation.json_validation import ModelHolderMixin


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


class MetadataMixin(ModelHolderMixin, Tablefyable):
    __metaclass__ = ABCMeta
    """
    Base object for entities that manage metadata. A MetadataEntity manages and id and a dict of metadata.
    The metadata should contain all necessary non-binary data to describe an entity.
    """

    METADATA = "metadata"
    CREATED_AT = 'created_at'
    UPDATED_AT = 'updated_at'
    LAST_MODIFIED_BY = 'last_modified_by'
    CREATED_BY = 'created_by'

    OVERWRITABLE = [CREATED_AT, CREATED_BY]

    @classmethod
    def _tablefy_register_columns(cls):
        # TODO make all fields tablefyable
        cls.tablefy_register("id", "name", cls.CREATED_AT, cls.CREATED_BY, cls.UPDATED_AT, cls.LAST_MODIFIED_BY)

    @abstractmethod
    def __init__(self, *, metadata: dict, **kwargs):

        import time

        metadata = {**{"id": uuid.uuid4().__str__(), self.CREATED_AT: time.time(), self.UPDATED_AT: time.time()}, **metadata}

        super().__init__(**{"metadata": metadata, **kwargs})


    @property
    def id(self):
        """
        returns the unique id of the data set. Data sets will be managed on the basis of this id
        :return: string
        """
        return self.metadata["id"]

    @id.setter
    def id(self, _id):
        """
        used for updating the id after the undlerying generic has assigned one
        :param _id: id, ideally an url
        :return:
        """
        self.metadata["id"] = _id

    @property
    def name(self):
        """
        returns the name of this object, which is expected in field "name" of the metadata. If this field does not
        exist, the id is returned
        :return:
        """
        if self.metadata and "name" in self.metadata:
            return self.metadata["name"]
        else:
            return str(self.id)

    @name.setter
    def name(self, name):
        self.metadata["name"] = name

    @property
    def created_at(self):
        if self.CREATED_AT in self.metadata:
            return self.metadata[self.CREATED_AT]
        else:
            return None

    @property
    def updated_at(self):
        if self.UPDATED_AT in self.metadata:
            return self.metadata[self.UPDATED_AT]
        else:
            return None

    @property
    def last_modified_by(self):
        if self.LAST_MODIFIED_BY in self.metadata:
            return self.metadata[self.LAST_MODIFIED_BY]
        else:
            return None

    @property
    def created_by(self):
        if self.CREATED_BY in self.metadata:
            return self.metadata[self.CREATED_BY]
        else:
            return None

    @property
    def metadata(self):
        return self._val_model

    def merge_metadata(self, metadata: dict):

        for key, value in metadata.items():
            # If the key is missing or key is to be overwritten
            if self.metadata.get(key, None) is None or key in self.OVERWRITABLE:
                self.metadata[key] = value
            else:
                pass


class ChildMixin(SuperStop):
    """ This is the abstract class being hierarchically nested in another class. This is relevant for app structure
    and the backend structure. For example the project backend is the parent of experiment backend """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, parent, **kwargs):
        self._parent = parent
        super().__init__(**kwargs)

    @property
    def parent(self):
        return self._parent
