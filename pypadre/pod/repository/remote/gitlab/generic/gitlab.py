"""
This file contains the implementation for
"""
# TODO: Handling of different file objects. It would be hard to keep track of all the file objects during an experiment
# TODO: Find a better way of mananging file and commit objects
# TODO: Create a dummy generic and check validity of all functions
# TODO: Lightweight function to validate the github repo and the git object along with the user
# NOTE: The gitlab api access token provides read/write access to the user.

import os
import uuid
from abc import abstractmethod, ABCMeta

import gitlab
from git import GitCommandError

from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.local.file.generic.i_git_repository import IGitRepository
from pypadre.pod.util.git_util import repo_exists, open_existing_repo, get_repo, add_and_commit


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
    def __init__(self, root_dir: str, gitlab_url:str, token:str ,backend: IPadreBackend,**kwargs):
        super().__init__(root_dir=root_dir,backend=backend,**kwargs)
        self._url = gitlab_url
        self._token = token
        self.authenticate()

    def authenticate(self):
        self._git = gitlab.Gitlab(self._url, private_token=self._token)

    def get_group(self,name):
        if self.group_exists(name):
            return self._git.groups.get(id=self._git.groups.list(search=name)[0].id)
        else:
            return self._git.groups.create({'name':name,'path':name})

    def group_exists(self,name):
        return len(self._git.groups.list(search=name))>0

    def get_projects(self, search_term):
        return self._git.projects.list(search=search_term) if self._git is not None else None

    def get_project_by_id(self, project_id, lazy=False):
        return self._git.projects.get(id=project_id, lazy=lazy) if self._git is not None else None

    def create_repo(self, name=""):
        if not self._repo_exists(name):
            try:
                if self._group:
                    self._repo = self._git.projects.create({'name': name,'namespace_id':self._group.id})
                else:
                    self._repo = self._git.projects.create({'name': name})
            except gitlab.GitlabCreateError as e:
                #TODO handle different exception upon creation
                pass
        else:
            self._repo = self.get_project_by_id(self.get_projects(name)[0].id)

    def _repo_exists(self, name):
        if self._group is not None:
            return len(self._group.projects.list(search=name))>0
        return len(self._git.projects.list(search=name))>0

    def get_repo_contents(self):
        return self._repo.repository_tree() if self._repo is not None else None

    def get_repo_sub_directory_contents(self, path, branch):
        return self._repo.repository_tree(path=path, ref=branch)

    def get_file_contents(self, path, branch, decode=True):
        # Get a file and print its content
        f = self._repo.files.get(file_path='README.rst', ref='master')
        # If decode flag is set, decode and return else return base64 encoded content
        return f.decode() if decode else f.content

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
            #TODO print warning
            raise ValueError("there is no remote generic. Create one")
        else:
            url= self.get_repo_url(ssh=ssh)
            _url = url.split("//")
            url = "".join([_url[0],"//","oauth2:{}@".format(self._token),_url[1]]) #To resolve the authentication https://stackoverflow.com/a/52154378
            return url

    def get_repo_url(self,ssh=False):
        return self._repo.attributes.get("ssh_url_to_repo") if ssh else self._repo.attributes.get("http_url_to_repo")

    def add_remote(self,branch,url):

        try:
            self._remote = self._local_repo.create_remote(branch, url)
        except GitCommandError as e:
            if "already exists" in e.stderr:
                self._remote = self._local_repo.remote(branch)

    def get(self,uid):
        """
        Gets the objects via uid. This might have to scan the metadatas on the remote repositories
        :param uid: uid to search for
        :return:
        """
        #TODO should we get the object from remote?
        return super().get(uid=uid)

    def put(self, obj, *args, merge=False, allow_overwrite=False, **kwargs):

        self.create_repo(name=obj.name)

        #TODO
        self._local_repo = super().put(obj)

        remote_url = self.get_remote_url()

        self.add_remote(self._branch,remote_url)

        directory = self.to_directory(obj)

        self._put(obj, *args, directory=directory,  merge=merge,**kwargs)

        self.reset()

    @abstractmethod
    def _put(self, obj, *args, directory: str,  merge=False, **kwargs):
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

    def list(self, search, offset=0, size=100):
        """

        :param search:
        :param offset:
        :param size:
        :return:
        """
        return super().list(search, offset,size)
        # repos = []
        # if self._group is None or "name" not in search.keys():
        #     return super().list(search)
        # else:
        #     name = search.get("name","") if search is not None else ""
        #     for repo in self._group.projects.list(search=name):
        #         repos.append(self._git.projects.get(repo.id,lazy=False))
        # return self.filter([self.get_by_repo(repo) for repo in repos],search)

    def get_file(self, repo, file: File):
        """
        Get a file in a generic by using a serializer name combination defined in a File object
        :param repo: Gitlab Repository object
        :param file: File object
        :return: Loaded file
        """
        if not isinstance(repo,gitlab.v4.objects.Project):
            return super().get_file(repo,file)
        try:
            f = repo.files.get(file_path=file.name, ref='master')
            data = file.serializer.deserialize(f.decode())
            return data
        except Exception as e:
            if self._local_repo is not None:
                return super().get_file(self._local_repo.working_dir, file)
            else:
                return None

    def update_file(self, file, content, branch, commit_message):
        # Update a file and if the file is binary, the calling function should serialize the content for modifying
        file.content = content
        file.save(branch=branch, commit_message=commit_message)

    def delete_file(self, file, commit_message):
        file.delete(commit_message=commit_message)

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

    def push_changes(self,commit_counter=0):
        if self._remote is None:
            remote_url = self.get_remote_url()
            self.add_remote(self._branch,remote_url)
        #TODO push if waiting commits is equal or more than the commit counter.
        try:
            self._remote.pull(refspec=self._branch)
            self._remote.push(refspec='{}:{}'.format(self._branch, self._branch))  # TODO commit/push schedule?
        except Exception as e:
            if "Couldn't find remote ref" in e.stderr:
                self._remote.push(refspec='{}:{}'.format(self._branch, self._branch))
            else:
                raise NotImplementedError

    @abstractmethod
    def update(self,*args):
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