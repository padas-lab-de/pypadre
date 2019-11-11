import uuid
from abc import ABCMeta, abstractmethod

from pypadre.core.printing.tablefyable import Tablefyable
from pypadre.core.util.inheritance import SuperStop
from pypadre.core.util.utils import _Const, is_jsonable
from pypadre.core.validation.validation import Validateable


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


class _CodeTypes(_Const):
    env = "environment"
    file = "file"
    fn = "function"


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


class MetadataEntity(Validateable, Tablefyable):
    __metaclass__ = ABCMeta
    """
    Base object for entities that manage metadata. A MetadataEntity manages and id and a dict of metadata.
    The metadata should contain all necessary non-binary data to describe an entity.
    """

    METADATA = "metadata"
    CREATED_AT = 'createdAt'
    UPDATED_AT = 'updatedAt'
    LAST_MODIFIED_BY = 'lastModifiedBy'
    CREATED_BY = 'createdBy'

    OVERWRITABLE = [CREATED_AT, CREATED_BY]

    @classmethod
    def _tablefy_register_columns(cls):
        cls.tablefy_register("id", "name", cls.CREATED_AT, cls.CREATED_BY, cls.UPDATED_AT, cls.LAST_MODIFIED_BY)

    @abstractmethod
    def __init__(self, *, metadata: dict, **kwargs):

        import time

        self._metadata = {**{"id": uuid.uuid4().__str__(), self.CREATED_AT: time.time(), self.UPDATED_AT: time.time()},
                          **metadata}

        # TODO remove this. This is only put here to find invalid / unserializable metadata without explicitly calling serialize on it
        if not is_jsonable(self._metadata):
            raise ValueError(str(self) + " is not json serializable!")

        # Merge all named parameters and kwargs together for validation
        # argspec = inspect.getargvalues(inspect.currentframe())
        # options = {**{key: argspec.locals[key] for key in argspec.args if key is not "self"}, **kwargs}

        super().__init__(**{self.METADATA: self._metadata, **kwargs})

        # if ontology_class is not None:
        #     # TODO validation json schema vs ontology itself?
        #     self._ontology_object = ontology_class(name=ontology_class.__name__ + "#" + self.id, **metadata)


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
        if self._metadata and "name" in self._metadata:
            return self._metadata["name"]
        else:
            return str(self.id)

    @name.setter
    def name(self, name):
        self._metadata["name"] = name

    @property
    def createdAt(self):
        if self.CREATED_AT in self._metadata:
            return self._metadata[self.CREATED_AT]
        else:
            return None

    @property
    def updatedAt(self):
        if self.UPDATED_AT in self._metadata:
            return self._metadata[self.UPDATED_AT]
        else:
            return None

    @property
    def lastModifiedBy(self):
        if self.LAST_MODIFIED_BY in self._metadata:
            return self._metadata[self.LAST_MODIFIED_BY]
        else:
            return None

    @property
    def createdBy(self):
        if self.CREATED_BY in self._metadata:
            return self._metadata[self.CREATED_BY]
        else:
            return None

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = metadata

    def merge_metadata(self, metadata: dict):

        for key, value in metadata.items():
            # If the key is missing or key is to be overwritten
            if self.metadata.get(key, None) is None or key in self.OVERWRITABLE:
                self.metadata[key] = value
            else:
                pass


class ChildEntity(SuperStop):
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
