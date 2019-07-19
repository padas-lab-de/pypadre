from abc import abstractmethod, ABCMeta

from git import Repo

from pypadre.backend.interfaces.backend.generic.i_base_meta_file_backend import IBaseMetaFileBackend


class IBaseGitBackend(IBaseMetaFileBackend):
    """ This is the abstract class implementation of a class extending the basic file backend with git functionality """
    __metaclass__ = ABCMeta
    # Variable holding the repository
    _repo = None
    _origin = None

    # Method to create a repository at the root directory of the object.
    # If bare=True, the function creates a bare bones repository
    @abstractmethod
    def create_bare_repo(self, bare=True):
        self.repo = Repo.init(self.root_dir, bare)

    @abstractmethod
    def create_remote(self, remote_name='origin', url=''):
        self.origin = self.repo.create_remote(name=remote_name, url=url)

    @abstractmethod
    def create_head(self, name):
        new_branch = self.repo.create_head(name)
        assert (self.repo.active_branch != new_branch)
        return new_branch

    @abstractmethod
    def create_tag(self, tag_name, ref_branch, message):
        tag = self.repo.create_tag(tag_name, ref=ref_branch, message=message)
        tag.commit

    @abstractmethod
    def create_sub_module(self, sub_repo_name, path_to_sub_repo, url, branch='master'):
        self.repo.create_submodule(sub_repo_name, path_to_sub_repo, url, branch)

    @abstractmethod
    def clone(self, url, path, branch='master'):
        if self.repo is not None:
            self.repo.clone_from(url, path, branch)

    @abstractmethod
    def _commit(self):
        self.repo.index.commit()

    @abstractmethod
    def get_untracked_files(self):
        return self.repo.untracked_files

    @abstractmethod
    def get_tags(self):
        return self.repo.tags

    @abstractmethod
    def get_working_tree_directory(self):
        return self.repo.working_tree_dir

    @abstractmethod
    def get_working_directory(self):
        return self.repo.working_dir

    @abstractmethod
    def get_git_path(self):
        return self.repo.git_dir

    @abstractmethod
    def is_head_remote(self):
        return self.repo.head.is_remote()

    @abstractmethod
    def is_head_valid(self):
        return self.repo.head.is_valid()

    @abstractmethod
    def get_heads(self):
        return self.repo.heads

    @abstractmethod
    def check_git_directory(self, path):
        return self.repo.git_dir.startswith(path)

    @abstractmethod
    def get_head(self):
        return self.repo.head

    @abstractmethod
    def delete_tags(self, tag_name):
        tags = self.repo.tags
        if tag_name in tags:
            self.repo.delete_tag(tag_name)

        else:
            # Raise warning/error that tag is not present
            pass

    @abstractmethod
    def archive_repo(self, path):
        with open(path, 'wb') as fp:
            self.repo.archive(fp)

    @abstractmethod
    def pull(self, name=None):
        origin = self.repo.remote(name=name)
        origin.pull()

    @abstractmethod
    def push(self, name=None):
        origin = self.repo.remote(name=name)
        origin.push()

    # database backend functions
    def list_datasets(self, search_id=None, search_metadata=None):
        """
        Function to list datasets. Will not be implemented for the Git Backend
        :param search_id:
        :param search_metadata:
        :return:
        """
        pass

    def put_dataset(self, dataset):
        pass

    def get_dataset(self, dataset_id, metadata_only=False):
        pass

    def delete_dataset(self, id):
        pass










