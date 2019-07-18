"""
This file contains the implementation for the GitHub backend to create and manage experiment repositories
"""
# TODO: Lightweight function to validate the github repo and the git object along with the user
from github import Github
from pypadre.backend.interfaces.backend.generic import i_base_git_backend
import uuid


class GitHubBackend(i_base_git_backend):

    _online_repo = None
    _user = None
    _git = None

    def _authenticate(self, name, password):
        self._git = Github(login_or_token=name, password=password)
        self._user = self.get_user()

    def create_repo(self, repo_name):
        self._online_repo = self._user.create_repo(repo_name)

    def get_user(self):
        return self._git.get_user()

    def create_file(self, path, comment, content, branch):
        t = self._online_repo.create_file(path, comment, content, branch=branch)
        return t

    def get_file_contents(self, path):
        return self._online_repo.get_contents(path)

    def delete_file(self, path, commit_message, commit_sha, branch):
        """
        Deletes a file from the GitHub online repository
        :param path: Path of the file in the repo
        :param commit_message: Commit message
        :param commit_sha: SHA of the commit
        :param branch: Branch where the file is to be deleted
        :return:
        """
        self._repo.delete_file(path, commit_message, commit_sha, branch=branch)

    def list_experiments(self, search_id=".*", search_metadata=None, start=-1, count=999999999,
                         remote=True):

        repo_list = []
        # Add all the repositories of the user to an array
        for repo in self._git.get_user(self._user.login).get_repos('all'):
            repo_list.append(repo.name)

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
        branch = "master"
        created_file = self.create_file(filename, comment, content, branch)


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
        # TODO Check if the experiment
        if experiment.experiment_configuration is not None:
            filename = "experiment.json"
            comment = "Creating experiment configuration at root experiment directory"
            content = self._metadata_serializer.serialise(experiment.metadata)
            branch = "master"
            created_file = self.create_file(filename, comment,
                                            self._metadata_serializer.serialise(experiment.experiment_configuration),
                                            branch)

    def put_run(self, experiment, run):
        """
        Stores a run of an experiment to the file repository.
        :param experiment: experiment the run is part of
        :param run: run to put
        :return:
        """
        if run.id is None:  # this is a new experiment
            run.id = uuid.uuid4()

        filename = '/'.join([run.id, "metadata.json"])
        comment = "Creating run directory"
        content = self._metadata_serializer.serialise(run.metadata)
        created_file = self.create_file(filename, comment,
                                        content,
                                        self._branch)

        filename = os.path.join([run.id, "hyperparameter.json"])
        comment = "Creating run directory"
        content = self._metadata_serializer.serialise(run.metadata)
        created_file = self.create_file(filename, comment,
                                        self._metadata_serializer.serialise(experiment.hyperparameters()),
                                        self._branch)

    def put_split(self, experiment, run, split):
        """
        Stores a run of an experiment to the file repository.
        :param experiment: experiment the run is part of
        :param run: run to put
        :return:
        """
        if split.id is None:  # this is a new experiment
            split.id = uuid.uuid4()

        filename = '/'.join([run.id, str(split.id), 'metadata.json'])
        comment = "Creating split directory"
        content = self._metadata_serializer.serialise(split.metadata)
        created_file = self.create_file(filename, comment, content, self._branch)

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
        created_file = self.create_file(filename, comment, content, self._branch)

    def put_metrics(self, experiment, run, split, metrics):
        """
        Writes the metrics of a split to the backend

        :param experiment: Experiment ID
        :param run: Run Id of the experiment
        :param split: Split ID
        :param metrics: dictionary containing all the required metrics to be written to the backend

        :return: None
        """

        filename = '/'.join([run.id, str(split.id), 'metrics.json'])
        comment = "Creating metrics file"
        content = self._metadata_serializer.serialise(metrics)
        created_file = self.create_file(filename, comment, content, self._branch)

    def log_end_experiment(self):

        filename = "log.txt"
        comment = "Ccompleting the experiment. Uploading the log file"
        content = ""

        created_file = self.create_file(filename, comment, content, self._branch)