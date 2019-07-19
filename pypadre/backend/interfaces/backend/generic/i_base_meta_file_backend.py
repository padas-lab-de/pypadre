import os
import shutil
from abc import abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_base_file_backend import IBaseFileBackend
from pypadre.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.backend.interfaces.backend.generic.i_storeable import IStoreable
from pypadre.backend.serialiser import JSonSerializer


class IBaseMetaFileBackend(IBaseFileBackend, ISearchable, IStoreable):
    """ This is the abstract class implementation of a basic metadata file store being able to store and to be searched. This backend
    has logic to write metadata files with a static name."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, parent, name, **kwargs):
        super().__init__(parent=parent, name=name)
        self._metadata_serializer = JSonSerializer

    META_FILE_NAME = "metadata.json"

    def put(self, obj):
        _dir = self.get_dir(self.to_folder_name(obj))
        try:
            with open(os.path.join(_dir, self.META_FILE_NAME), 'w') as f:
                metadata = dict(obj.metadata)
                f.write(self._metadata_serializer.serialise(metadata))
        except Exception as e:
            shutil.rmtree(_dir)
            raise e

    # Directory methods

    def get_meta_file_folder_name(self, folder_name):
        return self.get_meta_file(self.get_dir(folder_name))

    def get_meta_file(self, directory):
        with open(os.path.join(directory, self.META_FILE_NAME), 'r') as f:
            metadata = self._metadata_serializer.deserialize(f.read())
        return metadata