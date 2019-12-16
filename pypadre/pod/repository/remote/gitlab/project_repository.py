import os

from pypadre.core.model.project import Project
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IProjectRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.local.file.generic.i_log_file_repository import ILogFileRepository
from pypadre.pod.repository.local.file.project_repository import ProjectFileRepository
from pypadre.pod.repository.remote.gitlab.generic.gitlab import GitLabRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer, YamlSerializer, TextSerializer
from pypadre.pod.util.git_util import add_and_commit

NAME = 'projects'
_GROUP = '_projects'
META_FILE = File("metadata.json", JSonSerializer)
MANIFEST_FILE = File("manifest.yml", YamlSerializer)
GIT_IGNORE = File(".gitignore", TextSerializer)
_gitignore = "experiments/"


class ProjectGitlabRepository(GitLabRepository, IProjectRepository, ILogFileRepository):

    @staticmethod
    def placeholder():
        return '{PROJECT_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(root_dir=os.path.join(backend.root_dir, NAME), gitlab_url=backend.url, token=backend.token
                         , backend=backend)
        self._file_backend = ProjectFileRepository(backend=backend)
        self._group = self.get_group(name=_GROUP)
        self._tsrc = {"repos": []}

    def _get_by_repo(self, repo, path=''):
        if repo is None:
            return None
        metadata = self.get_file(repo, META_FILE)
        return Project(name=metadata.pop("name"), description=metadata.pop("description"), metadata=metadata)

    def _get_by_dir(self, directory):
        return self._file_backend._get_by_dir(directory=directory)

    def to_folder_name(self, project):
        # TODO only name for folder okay? (maybe a uuid, a digest of a config or similar?)
        return project.name

    def update(self, project: Project, src, url, commit_message: str):
        self._tsrc = self.get_file(self._repo, MANIFEST_FILE)
        self._tsrc["repos"].append({"src": src, "url": url})
        self.write_file(self.to_directory(project), MANIFEST_FILE, self._tsrc)
        add_and_commit(self.to_directory(project), message=commit_message, force_commit=True)
        self.push_changes()

    def _put(self, obj, *args, directory: str, merge=False, local=True, **kwargs):
        project = obj
        if merge:
            metadata = self.get_file(directory, META_FILE)
            if metadata is not None:
                project.merge_metadata(metadata)
        if local:
            self.write_file(directory, META_FILE, project.metadata)
            self.write_file(directory, MANIFEST_FILE, self._tsrc)
            self.write_file(directory, GIT_IGNORE, _gitignore)
            add_and_commit(directory, message="Adding the metadata and the manifest file of the project")

        if self.has_remote_backend(project):
            add_and_commit(directory, message="Adding unstaged changes in the repo")
            self.push_changes()
