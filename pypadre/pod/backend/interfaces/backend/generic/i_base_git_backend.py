import os
import platform
from abc import abstractmethod, ABCMeta

from git import Repo

from pypadre.pod.backend.interfaces.backend.generic.i_base_file_backend import FileBackend

"""
For datasets, experiments and projects there would be separate repositories.
The Dataset, experiment and project classes implement the IBaseGitBackend
So, the only functionalities required by git are add_file, list_file, delete_file, get_file
"""


class GitBackend(FileBackend):
    """ This is the abstract class implementation of a class extending the basic file backend with git functionality """
    __metaclass__ = ABCMeta
    # Variable holding the repository
    _remote_url = None
    _remote_name = 'remote'
    _remote = None
    _DEFAULT_GIT_MSG = 'Added file to git'

    # Method to create a repository at the root directory of the object.
    # If bare=True, the function creates a bare bones repository
    def __init__(self, *args, **kwargs):
        self._remote_url = kwargs.get('remote_url', None)
        self._remote_name = kwargs.get('remote_name', 'remote')

        super().__init__(*args, **kwargs)

    def _create_repo(self, path, bare=True):
        """
        Creates a local repository
        :param bare: Creates a bare git repository
        :return: Repo object
        """
        return Repo.init(path, bare)

    def _create_remote(self, repo, remote_name, url=''):
        """
        :param repo: The repo object that has to be passed
        :param remote_name: Name of the remote repository
        :param url: URL to the remote repository
        :return:
        """
        return repo.create_remote(name=remote_name, url=url)

    def _create_head(self, repo, name):
        """
        Creates a new branch
        :param name: Name of the new branch
        :return: Object to the new branch
        """
        new_branch = repo.create_head(name)
        assert (repo.active_branch != new_branch)
        return new_branch

    def _create_tag(self, repo, tag_name, ref_branch, message):
        """
        Creates a new tag for the branch
        :param repo: Repo object where the tag has to be created
        :param tag_name: Name for the tag
        :param ref_branch: Branch where the tag is to be created
        :param message: Message for the tag
        :return:
        """
        tag = repo.create_tag(tag_name, ref=ref_branch, message=message)
        tag.commit

    def _create_sub_module(self, repo, sub_repo_name, path_to_sub_repo, url, branch='master'):
        """
        Creating a submodule
        :param repo: Repo object where the submodule has to be created
        :param sub_repo_name: Name for the sub module
        :param path_to_sub_repo: Path to the submodule
        :param url: URL of the remote repo
        :param branch:
        :return:
        """
        repo.create_submodule(sub_repo_name, path_to_sub_repo, url, branch)

    def _clone(self, repo, url, path, branch='master'):
        """
        Clone a remote repo
        :param repo: Repo object of the repository
        :param url: URL of the remote remo
        :param path: Path to clone the remote repo
        :param branch: Branch to pull from the remote repo
        :return: None
        """
        if self.repo is not None:
            repo.clone_from(url, path, branch)

    def _commit(self, repo, message):
        """
        Commit a repository
        :param repo: Repo object
        :param message: Message when committing
        :return:
        """
        repo.git.commit(message=message)

    def _add_files(self, repo, file_path):
        """
        Adds the untracked files to the git
        :param file_path: An array containing the file paths to be added to git
        :return:
        """
        if self.is_backend_valid():
            if isinstance(file_path, str):
                repo.index.add([file_path])
            else:
                repo.index.add(file_path)


    def _get_untracked_files(self, repo):
        return repo.untracked_files if self.is_backend_valid() else None

    def _get_tags(self, repo):
        return repo.tags if self.is_backend_valid() else None

    def _get_working_tree_directory(self, repo):
        return repo.working_tree_dir if self.is_backend_valid() else None

    def _get_working_directory(self, repo):
        return repo.working_dir if self.is_backend_valid() else None

    def _get_git_path(self, repo):
        return repo.git_dir if self.is_backend_valid() else None

    def _is_head_remote(self, repo):
        return repo.head.is_remote() if self.is_backend_valid() else None

    def _is_head_valid(self, repo):
        return repo.head.is_valid() if self.is_backend_valid() else None

    def _get_heads(self, repo):
        return repo.heads if self.is_backend_valid() else None

    def _check_git_directory(self, repo, path):
        return repo.git_dir.startswith(path) if self.is_backend_valid() else None

    def _get_head(self, repo):
        return repo.head if self.is_backend_valid() else None

    def _has_uncommitted_files(self, repo):
        # True if there are files with differences
        return True if len([item.a_path for item in repo.index.diff(None)]) > 0 else False

    def _has_untracked_files(self, repo):
        return True if self._get_untracked_files(repo=repo) is not None else False

    def _add_untracked_files(self, repo):
        if self._has_untracked_files(repo=repo):
            untracked_files = self._get_untracked_files(repo=repo)
            self._add_files(repo=repo, file_path=untracked_files)

    def _delete_tags(self, repo, tag_name):
        if not self.is_backend_valid():
            return

        tags = repo.tags
        if tag_name in tags:
            repo.delete_tag(tag_name)

        else:
            # Raise warning/error that tag is not present
            pass

    def _archive_repo(self, path):
        if not self.is_backend_valid():
            return

        with open(path, 'wb') as fp:
            self.repo.archive(fp)

    def _pull(self, name=None):
        origin = self.repo.remote(name=name)
        origin.pull()

    def _push(self):
        if self._remote is None:
            self._remote = self._repo.create_remote(self._remote_name, self._remote_url)

        # Push to the master branch from current master branch
        # https://gitpython.readthedocs.io/en/stable/reference.html#git.remote.Remote.push
        self._remote.push(refspec='{}:{}'.format('master', 'master'))

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
    @abstractmethod
    def put(self, object, *args):
        super().put(object, args)

    def get(self, uid):
        # Call the File backend get function
        return super().get(uid=uid)

    def get_repo(self, path=None, url=None,  **kwargs):
        """
        Pull a repository from remote
        :param repo_name: Name of the repo to be cloned
        :param path: Path to be cloned
        :return:
        """
        if path is not None and url is not None:
            return Repo.clone_from(url=url, to_path=path)

        elif url is None and path is not None:
            # Open the local repository
            return Repo(path)

        super().get(**kwargs)

    def delete(self, id):
        """
        Deletes a repo from the Git backend
        :param id: id of the repo to be deleted
        :return:
        """
        # TODO: Remove the local directory
        # TODO: User will have to remove the remote repository by themselves
        super().delete(id)

    def is_backend_valid(self):
        """
        Check if repo is instantiated
        :return: True if valid, False otherwise
        """
        # TODO Implement validity checks for repo
        return True

    def has_remote_backend(self):
        # TODO Validate the remote_url
        return True if self.remote_url is not None else False

    def add_git_lfs_attribute_file(self, directory, file_extension):
        # Get os version and write content to file
        path = None
        # TODO: Verify path in Windows
        if platform.system() == 'Windows':
            path = os.path.join(directory, self.GIT_ATTRIBUTES)
        else:
            path = os.path.join(directory, self.GIT_ATTRIBUTES)

        f = open(path, "w")
        f.write(" ".join([file_extension, 'filter=lfs diff=lfs merge=lfs -text']))
        repo = self._create_repo(path=directory, bare=False)
        self._add_files(repo, file_path=path)
        self._commit(repo=repo, message='Added .gitattributes file for Git LFS')

        # Add all untracked files
        self._add_untracked_files(repo=repo)
        self._commit(repo, message=self._DEFAULT_GIT_MSG)

    @property
    def remote_name(self, remote_name):
        self._remote_name = remote_name

    def remote_name(self):
        return self._remote_name

    @property
    def remote_url(self, remote_url):
        self._remote_url = remote_url











