import os
from abc import abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_base_meta_file_backend import IBaseMetaFileBackend
from pypadre.backend.serialiser import PickleSerializer


class IBaseBinaryFileBackend(IBaseMetaFileBackend):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, parent, name, **kwargs):
        super().__init__(parent, name, **kwargs)
        self._data_serializer = PickleSerializer

    def has_binary_file(self, name):
        return os.path.exists(os.path.join(self.get_dir(name), "data.bin"))

    def get_binary_file(self, name):
        return self.get_binary_file_fn(name)()

    def get_binary_file_fn(self, name):
        def __load_data():
            with open(os.path.join(self.get_dir(name), "data.bin"), 'rb') as f:
                data = self._data_serializer.deserialize(f.read())
            return data
        return __load_data
