from abc import abstractmethod, ABCMeta

from git import Repo

from pypadre.backend.interfaces.backend.generic.i_base_meta_file_backend import IBaseMetaFileBackend

"""
For datasets, experiments and projects there would be separate repositories.
The Dataset, experiment and project classes implement the IBaseGitBackend
So, the only functionalities required by git are add_file, list_file, delete_file, get_file
"""


class IBaseGitBackend(IBaseMetaFileBackend):
    """ This is the abstract class implementation of a class extending the basic file backend with git functionality """
    __metaclass__ = ABCMeta
    # Variable holding the repository
    _repo = None
    _origin = None
    _REMOTE_URL = None
    _REMOTE_NAME = None
    _remote = None
    _DEFAULT_GIT_MSG = 'Added file to git'

    # Method to create a repository at the root directory of the object.
    # If bare=True, the function creates a bare bones repository

    def _create_repo(self, bare=True):
        """
        Creates a local repository
        :param bare: Creates a bare git repository
        :return:
        """
        self.repo = Repo.init(self._root_dir, bare)

    def _create_remote(self, remote_name='origin', url=''):
        """

        :param remote_name: Name of the remote repository
        :param url: URL to the remote repository
        :return:
        """
        self.origin = self.repo.create_remote(name=remote_name, url=url)

    def _create_head(self, name):
        """
        Creates a new branch
        :param name: Name of the new branch
        :return: Object to the new branch
        """
        new_branch = self.repo.create_head(name)
        assert (self.repo.active_branch != new_branch)
        return new_branch

    def _create_tag(self, tag_name, ref_branch, message):
        """
        Creates a new tag for the branch
        :param tag_name: Name for the tag
        :param ref_branch: Branch where the tag is to be created
        :param message: Message for the tag
        :return:
        """
        tag = self.repo.create_tag(tag_name, ref=ref_branch, message=message)
        tag.commit

    def _create_sub_module(self, sub_repo_name, path_to_sub_repo, url, branch='master'):
        """
        Creating a submodule
        :param sub_repo_name: Name for the sub module
        :param path_to_sub_repo: Path to the submodule
        :param url: URL of the remote repo
        :param branch:
        :return:
        """
        self.repo.create_submodule(sub_repo_name, path_to_sub_repo, url, branch)

    def _clone(self, url, path, branch='master'):
        """
        Clone a remote repo
        :param url: URL of the remote remo
        :param path: Path to clone the remote repo
        :param branch: Branch to pull from the remote repo
        :return: None
        """
        if self.repo is not None:
            self.repo.clone_from(url, path, branch)

    def _commit(self, message):
        self.repo.git.commit(message=message)

    def _add_files(self, file_path):
        """
        Adds the untracked files to the git
        :param file_path: An array containing the file paths to be added to git
        :return:
        """
        if self._repo is None:
            return

        self._repo.index.add([file_path])

    def _get_untracked_files(self):
        return self.repo.untracked_files

    def _get_tags(self):
        return self.repo.tags

    def _get_working_tree_directory(self):
        return self.repo.working_tree_dir

    def _get_working_directory(self):
        return self.repo.working_dir

    def _get_git_path(self):
        return self.repo.git_dir

    def _is_head_remote(self):
        return self.repo.head.is_remote()

    def _is_head_valid(self):
        return self.repo.head.is_valid()

    def _get_heads(self):
        return self.repo.heads

    def _check_git_directory(self, path):
        return self.repo.git_dir.startswith(path)

    def _get_head(self):
        return self.repo.head

    def _delete_tags(self, tag_name):
        tags = self.repo.tags
        if tag_name in tags:
            self.repo.delete_tag(tag_name)

        else:
            # Raise warning/error that tag is not present
            pass

    def _archive_repo(self, path):
        with open(path, 'wb') as fp:
            self.repo.archive(fp)

    def _pull(self, name=None):
        origin = self.repo.remote(name=name)
        origin.pull()

    def _push(self):
        if self._remote is None:
            self._remote = self._repo.create_remote(self._REMOTE_NAME, self._REMOTE_URL)

        # Push to the master branch from current master branch
        # https://gitpython.readthedocs.io/en/stable/reference.html#git.remote.Remote.push
        self._remote.push(refspec='{}:{}'.format('master', 'master'))

    # database backend functions
    def list(self, search_id=None, search_metadata=None):
        """
        Function to list repos. Cannot be implemented in GitPython.

        :param search_id:
        :param search_metadata:
        :return:
        """
        # TODO: Possibly, look for in the remote repositories if possible
        super().list(search_id, search_metadata)

    def put(self, object):
        """
        Writes the object to the git backend. If necessary, will create a new repo
        :param object: Object to be written to the repo
        :return:
        """
        if self._repo is None:
            self._create_repo(bare=False)

        # Write the files to the disk
        super().put(object)
        # Add the untracked file to the git
        self._add_files()
        # Call git commit
        self._repo.commit(message=self._DEFAULT_GIT_MSG)

    def get(self, path=None, url=None,  **kwargs):
        """
        Pull a repository from remote
        :param repo_name: Name of the repo to be cloned
        :param path: Path to be cloned
        :return:
        """
        #repo_url = '/'.join(self._REMOTE_URL, repo_name)
        if path is not None and url is not None:
            self._repo = Repo.clone_from(url=url, to_path=path)

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











