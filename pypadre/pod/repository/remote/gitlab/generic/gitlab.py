"""
This file contains the implementation for
"""
# TODO: Handling of different file objects. It would be hard to keep track of all the file objects during an experiment
# TODO: Find a better way of mananging file and commit objects
# TODO: Create a dummy generic and check validity of all functions
# TODO: Lightweight function to validate the github repo and the git object along with the user
# NOTE: The gitlab api access token provides read/write access to the user.

from abc import abstractmethod, ABCMeta
from logging import warning

import gitlab
from git import GitCommandError

from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.generic.i_repository_mixins import ISearchable
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.local.file.generic.i_git_repository import IGitRepository
from pypadre.pod.util.git_util import repo_exists, get_repo, crawl_repo

permissions = {"guest": gitlab.GUEST_ACCESS,
               "maintainer": gitlab.MAINTAINER_ACCESS,
               "developer": gitlab.DEVELOPER_ACCESS,
               "reporter": gitlab.REPORTER_ACCESS}


class GitLabRepository(IGitRepository):
    """ This is the abstract class extending the basic git backend with gitlab remote server functionality"""
    __metaclass__ = ABCMeta
    _repo = None
    _local_repo = None
    _remote = None
    _git = None
    _branch = "master"
    _group = None

    @abstractmethod
    def __init__(self, root_dir: str, gitlab_url: str, token: str, backend: IPadreBackend, **kwargs):
        super().__init__(root_dir=root_dir, backend=backend, **kwargs)
        self._url = gitlab_url
        self._token = token
        self.authenticate()
        self._user = self._git.user

    def authenticate(self):
        self._git = gitlab.Gitlab(self._url, private_token=self._token)
        self._git.auth()

    def git_env(self):
        return {'user': self._user.username, 'password': self._token}

    def get_group(self, name):
        if self.group_exists(name):
            return self._git.groups.get(id=self._git.groups.list(search=name)[0].id)
        else:
            return self._git.groups.create({'name': name, 'path': name})

    def group_exists(self, name):
        return len(self._git.groups.list(search=name)) > 0

    def get_projects(self, search_term):
        return self._git.projects.list(search=search_term) if self._git is not None else None

    def get_project_by_id(self, project_id, lazy=False):
        return self._git.projects.get(id=project_id, lazy=lazy) if self._git is not None else None

    def create_repo(self, obj):
        name = obj.name
        if not self._repo_exists(name):
            try:
                if self._group:
                    self._repo = self._git.projects.create({'name': name, 'namespace_id': self._group.id})
                else:
                    self._repo = self._git.projects.create({'name': name})

                self._local_repo = super().put(obj)
            except Exception as e:
                # TODO handle different exceptions upon creation
                print(e)
                pass
        else:
            self._repo = self.get_project_by_id(self.get_projects(name)[0].id)
            if not repo_exists(self.to_directory(obj)):
                self._local_repo = get_repo(path=self.to_directory(obj), url=self.get_remote_url())
            else:
                self._local_repo = get_repo(path=self.to_directory(obj))

    def _repo_exists(self, name):
        if self._group is not None:
            return len(self._group.projects.list(search=name)) > 0
        return len(self._git.projects.list(search=name)) > 0

    def get_repo_contents(self):
        return self._repo.repository_tree() if self._repo is not None else None

    def get_repo_sub_directory_contents(self, path, branch):
        return self._repo.repository_tree(path=path, ref=branch)

    def create_file(self, path, branch, content, email=None, name=None, encoding="text", commit_message="None"):
        f = self._repo.files.create({'file_path': path,
                                     'branch': branch,
                                     'content': content,
                                     'author_email': email,
                                     'author_name': name,
                                     'encoding': encoding,
                                     'commit_message': commit_message})
        return f

    def get_remote_url(self, ssh=False):
        if self._repo is None:
            # TODO print warning
            raise ValueError("there is no remote generic. Create one")
        else:
            url = self.get_repo_url(self._repo, ssh=ssh)
            return self.url_oauth(url)

    def url_oauth(self, url):
        _url = url.split("//")
        url = "".join([_url[0], "//", "oauth2:{}@".format(self._token),
                       _url[1]])  # To resolve the authentication https://stackoverflow.com/a/52154378
        return url

    def get_repo_url(self, repo=None, ssh=False):
        if repo is not None:
            return repo.attributes.get("ssh_url_to_repo") if ssh else self._repo.attributes.get("http_url_to_repo")
        return self._repo.attributes.get("ssh_url_to_repo") if ssh else self._repo.attributes.get("http_url_to_repo")

    def add_remote(self, branch, url):

        try:
            self._remote = self._local_repo.create_remote(branch, url)
        except GitCommandError as e:
            if "already exists" in e.stderr:
                self._remote = self._local_repo.remote(branch)

    def get(self, uid, rpath='', caller=None):
        """
        Gets the objects via uid. This might have to scan the metadatas on the remote repositories
        :param caller: the child backend calling the method
        :param rpath: relative pseudo-path in the repository
        :param uid: uid to search for
        :return:
        """
        # TODO should we get the object from remote?
        # return super().get(uid=uid)
        (repo, path) = self.find_repo_by_id(uid, rpath=rpath, caller=caller)
        if repo is None:
            return super().get(uid=uid)
        return self.get_by_repo(repo, path=path, caller=caller)
        # return super().get(uid=uid)

    def has_repo_dir(self, repo, path=None):
        return len(repo.repository_tree(path=path)) > 0

    def get_by_repo(self, repo, path='', caller=None):
        """

        :param caller:
        :param repo: repository of the object
        :param path: path to the object
        :return: the object
        """
        if not self.has_repo_dir(repo, path):
            return None

        try:
            if caller and hasattr(caller, '_get_by_repo'):
                return caller._get_by_repo(repo, path=path)
            return self._get_by_repo(repo, path=path)
        except ValueError:
            warning("Couldn't load object in repository " + str(repo.name) + ". Object might be corrupted.")

    def find_repo_by_id(self, uid, rpath='', caller=None):
        """
        Find a repo by searching for the corresponding id of the object.
        :param rpath: relative path to the object
        :param uid: Id to search for
        :return: (repo,path to the object)
        """
        repos = self.get_repos_by_search({'id': uid}, rpath=rpath, caller=caller)
        if len(repos) > 1:
            raise RuntimeError("Found multiple repositories for one ID. Data corrupted! " + str(repos))
        return repos.pop() if len(repos) == 1 else None

    def to_repo(self, obj):
        """
        Returns the repo of the object
        :param obj:
        :return:
        """
        # TODO find an efficient way to find the repo of the object
        return self._repo

    def get_repos_by_search(self, search, rpath="", caller=None):
        """
        Get a list of repositories depending on a search object
        :param search:
        :return: List of directories validated for the search
        """

        if self._group is None:
            return []
        else:
            if search is not None:
                repos = []
                for repo in self._group.projects.list():

                    repo = self._git.projects.get(repo.id)
                    if rpath == "":
                        repos.append((repo, ""))
                    else:
                        _path = ""
                        paths = crawl_repo(repo, rpath, _path)
                        if len(paths) != 0:
                            for path in paths:
                                repos.append((repo, path))
                return self.filter_repos(repos, search, caller=caller)
            else:
                return []

    def filter_repos(self, repos: list, search: dict, caller=None):
        if search is None:
            return repos
        return [(repo, path) for repo, path in repos if
                ISearchable.in_search(self.get_by_repo(repo, path=path, caller=caller), search)]

    def list(self, search, offset=0, size=100, caller=None):
        """

        :param caller:
        :param search:
        :param offset:
        :param size:
        :return:
        """

        rpath = ""
        if search is not None and 'rpath' in search:
            rpath = search.pop('rpath')
        repos = self.get_repos_by_search(search, rpath=rpath, caller=caller)
        objs = [self.get_by_repo(repo, path=path, caller=caller) for repo, path in repos]
        local_objs = []
        if hasattr(self, "_file_backend"):
            local_objs = self._file_backend.list(search, offset, size)
        return objs + local_objs

    def put(self, obj, *args, merge=False, allow_overwrite=False, **kwargs):

        self.create_repo(obj=obj)

        self.add_remote(self._branch, self.get_remote_url())

        self._put(obj, *args, directory=self.to_directory(obj), merge=merge, **kwargs)

        self.reset()

    @abstractmethod
    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        """
        This function pushes the files to the given remote branch from the local git repo.
        :param obj:
        :param args:
        :param directory:
        :param remote:
        :param merge:
        :param kwargs:
        :return:
        """
        pass

    def get_file(self, repo, file: File, default=None, path=''):
        """
        Get a file in a generic by using a serializer name combination defined in a File object
        :param path: path to the object folder
        :param default:
        :param repo: Gitlab Repository object
        :param file: File object
        :return: Loaded file
        """
        if not isinstance(repo, gitlab.v4.objects.Project):
            return super().get_file(repo, file, default=default)
        try:
            file_path = path + '/' + file.name if path != '' else file.name
            f = repo.files.get(file_path=file_path, ref='master')
            data = file.serializer.deserialize(f.decode())
            return data
        except Exception as e:
            return None

    def add_member(self, user="", permission="guest"):
        user = self.get_user(user=user)
        if user is not None:
            try:
                self._repo.members.create({'user_id': user.id, 'access_level': permissions.get(permission)})
            except Exception as e:
                raise Exception

    def remove_member(self, user=""):
        user = self.get_user(user=user)
        try:
            self._repo.members.delete(user.id)
        except Exception as e:
            raise Exception

    def get_user(self, user=""):
        users = self._git.users.list(username=user)
        if len(users) == 1:
            return self._git.users.get(users.pop().id)
        else:
            raise ValueError("username is ambiguous or is not existing")

    def commit(self, **options):
        # Create a commit
        # See https://docs.gitlab.com/ce/api/commits.html#create-a-commit-with-multiple-files-and-actions
        # for actions detail
        """
        data = {
            'branch_name': 'master',  # v3
            'branch': 'master',  # v4
            'commit_message': 'blah blah blah',
            'actions': [
                {
                    'action': 'create',
                    'file_path': 'README.rst',
                    'content': open('path/to/file.rst').read(),
                },
                {
                    # Binary files need to be base64 encoded
                    'action': 'create',
                    'file_path': 'logo.png',
                    'content': base64.b64encode(open('logo.png').read()),
                    'encoding': 'base64',
                }
            ]
        }
        """
        commit = self._repo.commits.create(options)
        return commit

    def push_changes(self, commit_counter=0):
        if self._remote is None:
            remote_url = self.get_remote_url()
            self.add_remote(self._branch, remote_url)
        # TODO push if waiting commits is equal or more than the commit counter.
        try:
            self._remote.pull(refspec=self._branch)
            self._remote.push(refspec='{}:{}'.format(self._branch, self._branch))  # TODO commit/push schedule?
        except Exception as e:
            if "Couldn't find remote ref" in e.stderr:
                self._remote.push(refspec='{}:{}'.format(self._branch, self._branch))
            else:
                raise Exception

    @abstractmethod
    def update(self, *args):
        pass

    def upload_file(self, filename, path):
        if self._repo is not None:
            self._repo.upload(filename, filepath=path)

    def reset(self):
        self._remote = None

    @property
    def remote(self):
        return self._remote

    def __del__(self):
        # close the gitlab session
        self._git.__exit__()
