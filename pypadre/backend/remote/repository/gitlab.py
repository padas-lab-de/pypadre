"""
This file contains the implementation for
"""
# TODO: Handling of different file objects. It would be hard to keep track of all the file objects during an experiment
# TODO: Find a better way of mananging file and commit objects
# TODO: Create a dummy repository and check validity of all functions
# TODO: Lightweight function to validate the github repo and the git object along with the user
# NOTE: The gitlab api access token provides read/write access to the user.

import gitlab
import base64
from pypadre.backend.interfaces.backend.generic.i_base_git_backend import IBaseGitBackend


class GitLabBackend(IBaseGitBackend):
    _repo = None
    _git = None

    def authenticate(self, url, private_token):
        self._git = gitlab.Gitlab(url, private_token=private_token)

    def get_projects(self, search_term):
        return self._git.projects.list(search=search_term) if self._git is not None else None

    def get_project_by_id(self, project_id, lazy=True):
        return self._git.projects.get(id=project_id, lazy=lazy) if self._git is not None else None

    def create_repo(self, name):
        self._repo = self._git.projects.create({'name': name})

    def get_repo_contents(self):
        return self._repo.repository_tree() if self._repo is not None else None

    def get_repo_sub_directory_contents(self, path, branch):
        return self._repo.repository_tree(path=path, ref=branch)

    def get_file_contents(self, path, branch, decode=True):
        # Get a file and print its content
        f = self._repo.files.get(file_path='README.rst', ref='master')
        # If decode flag is set, decode and return else return base64 encoded content
        return f.decode() if decode else f.content

    def create_file(self, path, branch, content, email, name, encoding, commit_message):
        f = self._repo.files.create({'file_path': path,
                                     'branch': branch,
                                     'content': content,
                                     'author_email': email,
                                     'author_name': name,
                                     'encoding': encoding,
                                     'commit_message': commit_message})
        return f

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

    def upload_file(self, filename, path):
        if self._repo is not None:
            self._repo.upload(filename, filepath=path)

