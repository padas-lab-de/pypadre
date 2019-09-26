import os
import re
import shutil
from abc import abstractmethod, ABCMeta

from pypadre.core.base import ChildEntity
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.generic.i_repository_mixins import IStoreableRepository, ISearchable, IRepository
from pypadre.pod.util.file_util import get_path


class File:
    def __init__(self, name, serializer):
        self._name = name
        self._serializer = serializer

    @property
    def name(self):
        """
        Files have a name in the directory structure
        :return: Name
        """
        return self._name

    @property
    def serializer(self):
        """
        A file defines its serializer for peristing purposes.
        :return: The serializer to use
        """
        return self._serializer


class IFileRepository(IRepository, ISearchable, IStoreableRepository):
    """ This is the abstract class implementation of a backend storing its information onto the disk in a file
    structure"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, root_dir: str, backend: IPadreBackend, **kwargs):
        self.root_dir = root_dir
        super().__init__(backend=backend, **kwargs)

    def get(self, uid):
        """
        Gets the object via uid. This might have to scan the metadatas on the local file system.
        :param uid: uid to search for
        :return:
        """
        return self.get_by_dir(self.find_dir_by_id(uid))

    def list(self, search, offset=0, size=100):
        """
        List all objects in the repository
        :param offset:
        :param size:
        :param search: search object. You can pass key value pairs to search for.
        """
        folder = ""
        if search is not None and 'folder' in search:
            folder = search.get('folder')
        dirs = self.find_dirs(folder)
        # TODO add offset und size
        return self.filter([self.get_by_dir(d) for d in dirs], search)

    def delete_by_id(self, uid):
        """
        Delete an object by id. This will call get to find the object to delete.
        :param uid: uid of the object to delete
        :return:
        """
        self.delete(self.get(uid))

    def delete(self, obj):
        """
        Delete an given object. This removes the file in the filesystem.
        :param obj: Object to delete
        :return:
        """
        self.delete_dir(self.to_directory(obj))

    def put(self, obj, *args, merge=False, allow_overwrite=False, **kwargs):
        """
        Put an object to the file backend
        :param allow_overwrite:
        :param *args:
        :param update:
        :param **kwargs:
        :param obj:
        :param args:
        :param kwargs:
        :return:
        """
        directory = self.to_directory(obj)

        # Create or overwrite folder
        if os.path.exists(directory):
            if not allow_overwrite:
                raise ValueError("Object path %s already exists.".format(obj) +
                                 "Overwriting not explicitly allowed. Set allow_overwrite=True")
            shutil.rmtree(directory)
        os.makedirs(directory)

        # Put data of the object
        self._put(obj, *args, directory=directory, merge=merge, **kwargs)

    @abstractmethod
    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        """
        This function writes the object to files in the given directory. This has to be implemented by the respective
        backend.
        :param obj:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def to_folder_name(self, obj):
        """
        This function will get a folder name to a given object.
        :param obj: Object to store in a folder
        :return: Name of the folder
        """
        return str(obj.id)

    @abstractmethod
    def get_by_dir(self, directory):
        """
        Gets an object for a given directory.
        :param directory: Directory to load the object from
        :return: Object which should be deserialized
        """
        pass

    def find_dir_by_id(self, uid):
        """
        Find a dir by searching for the corresponding id of the object.
        :param uid: Id to search for
        :return: Directory of the object
        """
        # TODO: Change the hardcoded 'id' to a key value to be searched
        dirs = self.get_dirs_by_search({'id': uid})
        return dirs.pop() if len(dirs) > 0 else []

    def has_dir(self, directory):
        """
        Checks if a directory with given folder name exists in the current root dir.
        :param directory: Name of the folder to check for
        :return: true if the directory exists
        """
        return os.path.exists(directory)

    def get_dirs_by_search(self, search):
        """
        Get a list of directories depending on a search object.
        :param search: The search object
        :return: List of directories validated for the search
        """
        return [self.to_directory(o) for o in self.list(search)]

    def find_dirs(self, matcher, strip_postfix=""):
        # TODO postfix stripping?
        dirs = self._get_all_dirs()
        # dirs = [f for f in os.listdir(self.root_dir) if f.endswith(strip_postfix)]

        if matcher is not None:
            rid = re.compile(matcher)
            dirs = [d for d in dirs if rid.match(d)]

        if len(strip_postfix) == 0:
            return dirs
        else:
            return [dir[:-1 * len(strip_postfix)] for dir in dirs
                    if dir is not None and len(dir) >= len(strip_postfix)]

    def delete_dir(self, directory):
        """
        :param directory: the folder name of the object to delete
        :return:
        """
        if self.has_dir(directory):
            shutil.rmtree(directory)

    def get_file(self, dir, file: File):
        """
        Get a file in a directory by using a serializer name combination defined in a File object.
        :param dir: Location of the repo
        :param file: File object
        :return: Loaded file
        """
        return self.get_file_fn(dir, file)()

    def has_file(self, dir, file: File):
        """
        Check if a file in a directory by using a serializer name combination defined in a File object exists.
        :param dir: Location of the repo
        :param file: File object
        :return: true if file exists
        """
        return os.path.exists(os.path.join(dir, file.name))

    def get_file_fn(self, dir, file: File):
        """
        Method to get a lazy loading function for the file.
        :param dir: Location of the repo
        :param file: File object
        :return: Function to load the file data
        """

        def __load_data():
            if not os.path.exists(os.path.join(dir, file.name)):
                # TODO Raise exception
                return None
            with open(os.path.join(dir, file.name), 'rb') as f:
                data = file.serializer.deserialize(f.read())
            return data

        return __load_data

    def write_file(self, dir, file: File, target, mode="w"):
        """
        Write given file object into directory with given name and serializer
        :param dir: directory
        :param file: file object containing name and serializer
        :param target: target to serialize
        :param mode: The mode to use when writing to disk
        :return:
        """
        with open(os.path.join(dir, file.name), mode) as f:
            f.write(file.serializer.serialise(target))

    def to_directory(self, obj):
        """
        Returns the path of the object
        :param obj:
        :return:
        """
        return self.replace_placeholder(obj, get_path(self.root_dir, self.to_folder_name(obj), False))

    @staticmethod
    @abstractmethod
    def placeholder():
        """
        # Every file backend should define a placeholder to represent an object id in a root directory template.
        :return: Placeholder for the path string
        """
        pass

    @classmethod
    def has_placeholder(cls):
        return cls.placeholder() is not None

    def replace_placeholder(self, obj, path):
        if not self.has_placeholder():
            return path
        return path.replace(self.placeholder(), self.to_folder_name(obj))

    def _replace_placeholders_with_wildcard(self, path):
        import re
        return re.sub("{.*?}", "*", path)

    def _get_all_dirs(self):
        import glob
        return glob.glob(self._replace_placeholders_with_wildcard(self.root_dir) + "/*")


class IChildFileRepository(IFileRepository, ChildEntity):

    @abstractmethod
    def __init__(self, *, parent: IFileRepository, name: str, backend: IPadreBackend, **kwargs):
        if parent.has_placeholder():
            rootdir = os.path.join(parent.root_dir, parent.placeholder(), name)
        else:
            rootdir = os.path.join(parent.root_dir, name)
        super().__init__(backend=backend, name=name, parent=parent, root_dir=rootdir,
                         **kwargs)

    def put(self, obj, *args, merge=False, allow_overwrite=False, **kwargs):

        # Put parent if it is not already existing
        if self.parent is not None and isinstance(obj, ChildEntity):
            if not self.parent.exists(obj.parent.id):
                self.parent.put(obj.parent)

        super().put(obj, *args, merge=merge, allow_overwrite=allow_overwrite, **kwargs)

    def replace_placeholder(self, obj, path):
        if self.placeholder() is not None and self.placeholder() in path:
            if hasattr(self.parent, 'replace_placeholder'):

                # Call the replace function of the basic file backend but also replace parent
                return self.parent.replace_placeholder(obj.parent, super().replace_placeholder(obj, path))
            else:

                # Call the replace function of the basic file backend
                return super().replace_placeholder(obj, path)

        # If no placeholder is present we can call the parent placeholder replacement function
        elif hasattr(self.parent, 'replace_placeholder'):
            return self.parent.replace_placeholder(obj.parent, path)
        return path

    def get_parent_dir(self, directory):
        return os.path.abspath(os.path.join(directory, '../..'))
