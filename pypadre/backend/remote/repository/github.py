"""
This file contains the implementation for the GitHub backend to create and manage experiment repositories
"""
# TODO: Lightweight function to validate the github repo and the git object along with the user
from github import Github
from pypadre.backend.interfaces.backend.generic import i_base_git_backend


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

