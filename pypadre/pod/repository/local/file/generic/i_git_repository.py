from abc import abstractmethod, ABCMeta

from git import Repo

from pypadre.pod.repository.local.file.generic.i_file_repository import IFileRepository
from pypadre.pod.util.git_util import repo_exists, open_existing_repo, get_repo, add_and_commit

"""
For datasets, experiments and projects there would be separate repositories.
The Dataset, experiment and project classes implement the IBaseGitRepository
So, the only functionalities required by git are add_file, list_file, delete_file, get_file
"""


class IGitRepository(IFileRepository):
    """ This is the abstract class implementation of a class extending the basic file backend with git functionality """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # database backend functions
    def list(self, search, offset=0, size=100):
        """
        Function to list repos. Cannot be implemented in GitPython.
        :param search_id: Id to be searched for
        :param offset:
        :param size:
        :return:
        """
        # TODO: Possibly, look for in the remote repositories if possible
        return super().list(search)

    # Abstract method which would create a repo based on the requirements
    def put(self, obj, *args, merge=False, allow_overwrite=False, **kwargs):
        super().put(obj, *args, merge=merge, allow_overwrite=allow_overwrite, **kwargs)

        # Init repo if not already existing
        directory = self.to_directory(obj)
        if not repo_exists(directory):
            repo = Repo.init(path=directory, **kwargs.pop("repo_kwargs", {}))
            add_and_commit(directory, message=kwargs.pop('message', 'Initial Commit of Repository'))
        else:
            repo = get_repo(path=directory, **kwargs.pop("repo_kwargs", {}))
            add_and_commit(directory, message=kwargs.pop('message', 'commiting existing changes'))

        return repo

    def get(self, uid):
        # Call the File backend get function
        return super().get(uid=uid)

    @abstractmethod
    def get_by_repo(self, repo, rpath='', caller=None):
        """
        Gets an object for a given generic.
        :param rpath: relative path in the repo
        :param repo: repository to load the object from
        :return: Object which should be deserialized
        """
        raise NotImplementedError

    @abstractmethod
    def has_repo_dir(self, repo, rpath=None):
        raise NotImplementedError

    @abstractmethod
    def _get_by_repo(self, repo, path=None):
        raise NotImplementedError

    def delete(self, id_):
        """
        Deletes a repo from the Git backend
        :param id_: id of the repo to be deleted
        :return:
        """
        # TODO: User will have to remove the remote generic by themselves
        super().delete(id_)

    @staticmethod
    def is_backend_valid():
        """
        Check if repo is instantiated
        :return: True if valid, False otherwise
        """
        # TODO Implement validity checks for repo
        return True

    def has_remote_backend(self, obj):

        # Get the directory path from the object
        dir_path = self.to_directory(obj)

        # Check if there is a generic existing in the path or any path of the parent directories
        repo = open_existing_repo(dir_path, search_parents=False)

        # If a generic does not exist return false
        if repo is None:
            return False

        return len(repo.remotes) > 0
