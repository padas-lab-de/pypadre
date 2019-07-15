import os
import re
import shutil
from abc import abstractmethod, ABCMeta

from pypadre.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.backend.interfaces.backend.generic.i_storeable import IStoreable
from pypadre.backend.interfaces.backend.generic.i_sub_backend import ISubBackend
from pypadre.backend.serialiser import PickleSerializer, JSonSerializer
from pypadre.util.file_util import get_path


class IBaseFileBackend(ISubBackend):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, parent, name, **kwargs):
        super().__init__(parent=parent)
        self.root_dir = os.path.join(self._parent.root_dir, name)

    def get(self, uid):
        return self.get_by_dir(self.find_dir_by_id(uid))

    def list(self, search):
        """
        List all objects in the repository
        :param search: search object. You can pass key value pairs to search for.
        """
        folder = ""
        if 'folder' in search:
            folder = search.get('folder')
        dirs = self.find_dirs(folder)
        return self.filter([self.get_by_dir(d) for d in dirs], search)

    def delete_by_id(self, uid):
        self.delete(self.get(uid))

    def delete(self, obj):
        self.delete_dir(self.to_folder_name(obj))

    # Directory methods
    @abstractmethod
    def to_folder_name(self, obj):
        pass

    @abstractmethod
    def get_by_dir(self, directory):
        pass

    def find_dir_by_id(self, uid):
        # TODO multiple match?
        return self.get_dirs_by_search({'id': uid}).pop()

    def has_dir(self, folder_name):
        return os.path.exists(self.get_dir(folder_name))

    def get_dirs_by_search(self, search):
        return [self.get_dir(self.to_folder_name(o)) for o in self.list(search)]

    def get_dir(self, folder_name):
        return get_path(self.root_dir, str(folder_name))

    def find_dirs(self, matcher, strip_postfix=""):
        files = [f for f in os.listdir(self.root_dir) if f.endswith(strip_postfix)]
        if matcher is not None:
            rid = re.compile(matcher)
            files = [f for f in files if rid.match(f)]

        if len(strip_postfix) == 0:
            return files
        else:
            return [file[:-1 * len(strip_postfix)] for file in files
                    if file is not None and len(file) >= len(strip_postfix)]

    def delete_dir(self, folder_name):
        """
        :param folder_name: the folder name of the object to delete
        :return:
        """
        if self.has_dir(folder_name):
            shutil.rmtree(self.get_dir(folder_name))