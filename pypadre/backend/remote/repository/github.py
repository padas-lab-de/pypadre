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




