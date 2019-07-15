import os
import shutil
from abc import abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_base_meta_file_backend import IBaseMetaFileBackend
from pypadre.backend.serialiser import PickleSerializer


class IBaseBinaryFileBackend(IBaseMetaFileBackend):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, parent, name, **kwargs):
        super().__init__(parent, name, **kwargs)
        self._data_serializer = PickleSerializer

    BINARY_FILE_NAME = "data.bin"

    def has_binary_file(self, directory):
        return os.path.exists(os.path.join(directory, self.BINARY_FILE_NAME))

    def get_binary_file(self, directory):
        return self.get_binary_file_fn(directory)()

    def get_binary_file_fn(self, directory):
        def __load_data():
            with open(os.path.join(directory, self.BINARY_FILE_NAME), 'rb') as f:
                data = self._data_serializer.deserialize(f.read())
            return data
        return __load_data

    def put(self, obj):
        super().put(obj)
        _dir = self.get_dir(self.to_folder_name(obj))
        try:
            if obj.has_data():
                with open(os.path.join(_dir, self.BINARY_FILE_NAME), 'wb') as f:
                    f.write(self._data_serializer.serialise(obj.data))
        except Exception as e:
            shutil.rmtree(_dir)
            raise e
