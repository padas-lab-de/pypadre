"""
This file contains the implementation for
"""
# TODO: Handling of different file objects. It would be hard to keep track of all the file objects during an experiment
# TODO: Find a better way of mananging file and commit objects
# TODO: Create a dummy repository and check validity of all functions
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
            raise ValueError("there is no remote repository. Create one")
        else:
            attributes = self._repo.attributes
            url= attributes.get("ssh_url_to_repo") if ssh else attributes.get("http_url_to_repo")
            _url = url.split("//")
            url = "".join([_url[0],"//","oauth2:{}@".format(self._token),_url[1]]) #To resolve the authentication https://stackoverflow.com/a/52154378
            return url

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
        super().get(uid=uid)

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
        repos = []
        if self._group is None:
            return super().list(search)
        else:
            for repo in self._group.projects.list(search=search.get("name")):
                repos.append(self._git.projects.get(repo.id,lazy=False))
        return self.filter([self.get_by_repo(repo) for repo in repos],search)

    def get_file(self, repo, file: File):
        """
        Get a file in a repository by using a serializer name combination defined in a File object
        :param repo: Gitlab Repository object
        :param file: File object
        :return: Loaded file
        """
        try:
            f = repo.files.get(file_path=file.name, ref='master')
            data = file.serializer.deserialize(f.decode())
            return data
        except gitlab.GitlabGetError as e:
            return super().get_file(repo,file)

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

    def push_changes(self):
        try:
            self._remote.pull(refspec=self._branch)
            self._remote.push(refspec='{}:{}'.format(self._branch, self._branch))  # TODO commit/push schedule?
        except Exception as e:
            if "Couldn't find remote ref" in e.stderr:
                self._remote.push(refspec='{}:{}'.format(self._branch, self._branch))
            else:
                return

    def upload_file(self, filename, path):
        if self._repo is not None:
            self._repo.upload(filename, filepath=path)

    def list_experiments(self, search_id=".*", search_metadata=None, start=-1, count=999999999,
                         remote=True):

        repo_list = []
        # Add all the repositories of the user to an array
        repo_list = self._git.projects.list()
        # TODO: Process the list

        return repo_list

    def delete_experiments(self, experiment=".*", mode="all", search_metadata=None):
        """
        Padre does not support removal of experiments on the GitHub
        :param experiment:
        :param mode:
        :param search_metadata:
        :return:
        """
        pass

    def put_experiment(self, experiment, append_runs=False, allow_overwrite=True):

        repo_exists = False
        # Check if repo already exists

        if not repo_exists:
            self.create_repo(experiment.name)

        filename = "metadata.json"
        comment = "Creating metadata.json at root experiment directory"
        content = self._metadata_serializer.serialise(experiment.metadata)
        created_file = self.create_file(filename, self._branch, content, None, self._user,
                                        'text', 'adding metadata.json')


    def get_experiment(self):
        """
        Clones the experiment from the github repository
        https://stackoverflow.com/questions/46937032/using-the-node-github-api-to-clone-a-remote-repo-locally
        :return:
        """
        # Use the local git package to clone the repo locally
        pass

    def put_experiment_configuration(self, experiment):
        """
        Serializes the experiment configuration for purposes of sharing
        :param experiment: The experiment object
        :return:
        """
        # TODO Check if the experiment.json file already exists in the repository
        if experiment.experiment_configuration is not None:
            filename = "experiment.json"
            comment = "Creating experiment configuration at root experiment directory"
            content = self._metadata_serializer.serialise(experiment.metadata)
            created_file = self.create_file(filename, comment,
                                            self._metadata_serializer.serialise(experiment.experiment_configuration),
                                            self._branch)

    def put_run(self, experiment, run):
        """
        Stores a run of an experiment to the file repository.
        :param experiment: experiment the run is part of
        :param run: run to put
        :return:
        """
        if run.uid is None:  # this is a new experiment
            run.uid = uuid.uuid4()

        filename = '/'.join([run.id, "metadata.json"])
        comment = "Creating run directory"
        content = self._metadata_serializer.serialise(run.metadata)
        created_file = self.create_file(filename, self._branch, content, None, self._user,
                                        'text', comment)

        filename = os.path.join([run.id, "hyperparameter.json"])
        comment = "Creating run directory"
        content = self._metadata_serializer.serialise(run.metadata)
        created_file = self.create_file(filename, self._branch, content, None, self._user,
                                        'text', comment)

    def put_split(self, experiment, run, split):
        """
        Stores a run of an experiment to the file repository.
        :param experiment: experiment the run is part of
        :param run: run to put
        :return:
        """
        if split.uid is None:  # this is a new experiment
            split.uid = uuid.uuid4()

        filename = '/'.join([run.id, str(split.id), 'metadata.json'])
        comment = "Creating split directory"
        content = self._metadata_serializer.serialise(split.metadata)
        created_file = self.create_file(filename, self._branch, content, None, self._user,
                                        'text', comment)

    def put_results(self, experiment, run, split, results):
        """
        Write the results of a split to the backend

        :param experiment: Experiment ID
        :param run_id: Run ID of the current experiment run
        :param split_id: Split id
        :param results: results to be written to the backend

        :return: None
        """
        filename = '/'.join([run.id, str(split.id), 'results.json'])
        comment = "Creating results file"
        content = self._metadata_serializer.serialise(results)
        created_file = self.create_file(filename, self._branch, content, None, self._user,
                                        'text', comment)

    def put_metrics(self, experiment, run, split, metrics):
        """
        Writes the metrics of a split to the backend

        :param experiment: Experiment ID
        :param run: Run Id of the experiment
        :param split: Split ID
        :param metrics: dictionary containing all the required metrics to be written to the backend

        :return: None
        """
        split_id = str(split.id)

        filename = '/'.join([run.id, str(split.id), 'metrics.json'])
        comment = "Creating metrics file"
        content = self._metadata_serializer.serialise(metrics)

        created_file = self.create_file(filename, self._branch, content, None, self._user,
                                        'text', comment)

    def log_end_experiment(self):

        filename = "log.txt"
        comment = "Completing the experiment. Uploading the log file"
        content = ""

        created_file = self.create_file(filename, self._branch, content, None, self._user,
                                        'text', comment)
    def reset(self):
        self._repo = None
        self._remote = None
        self._local_repo = None

    @property
    def remote(self):
        return self._remote

    def __del__(self):
        #close the gitlab session
        self._git.__exit__()