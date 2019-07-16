"""
This file contains the implementation for the GitHub backend to create and manage experiment repositories
"""
from github import Github
from pypadre.backend.interfaces.backend.generic import i_base_git_backend


class GitHubBackend(i_base_git_backend):

    _online_repo = None

    def _authenticate(self, url, private_token):
        pass

