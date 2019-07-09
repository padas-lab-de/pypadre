import os
from abc import abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_sub_backend import ISubBackend
from pypadre.backend.serialiser import PickleSerializer, JSonSerializer
from pypadre.util.file_util import get_path


class IBaseMetaFileBackend(ISubBackend):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, parent, name, **kwargs):
        super().__init__(parent=parent)
        self._metadata_serializer = JSonSerializer
        self.root_dir = os.path.join(self._parent.root_dir, name)

    def get_dir(self, name):
        return get_path(self.root_dir, str(name))

    def get_meta_file(self, name):
        with open(os.path.join(self.get_dir(name), "metadata.json"), 'r') as f:
            metadata = self._metadata_serializer.deserialize(f.read())
        return metadata
