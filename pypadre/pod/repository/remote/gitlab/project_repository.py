import os
import re

from pypadre.core.model.project import Project
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IProjectRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.remote.gitlab.repository.gitlab import GitLabRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer, YamlSerializer, TextSerializer
from pypadre.pod.util.git_util import add_and_commit

NAME = 'projects'
_GROUP = '_projects'
META_FILE = File("metadata.json", JSonSerializer)
MANIFEST_FILE = File("manifest.yml", YamlSerializer)
GIT_IGNORE = File(".gitignore", TextSerializer)
_gitignore = "experiments/"

class ProjectGitlabRepository(GitLabRepository, IProjectRepository):

    @staticmethod
    def placeholder():
        return '{PROJECT_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(root_dir=os.path.join(backend.root_dir, NAME),gitlab_url=backend.url,token=backend.token
                         , backend=backend)
        self._group = self.get_group(name=_GROUP)
        self._tsrc = {"repos": []}

    def get_by_repo(self, repo):
        metadata = self.get_file(repo, META_FILE)
        return Project(name=metadata.pop("name"), description=metadata.pop("description"), metadata=metadata)

    def to_folder_name(self, project):
        # TODO only name for folder okay? (maybe a uuid, a digest of a config or similar?)
        return "{}".format(project.name)

    def get_by_name(self, name):
        """
        Shortcut because we know name is the folder name. We don't have to search in metadata.json
        :param name: Name of the dataset
        :return:
        """
        return self.list({'folder': re.escape(name)})

    def _put(self, project: Project, *args, directory: str, merge=False, **kwargs):
        #TODO tsrc manifest file creation
        if merge:
            metadata = self.get_file(directory, META_FILE)
            if metadata is not None:
                project.merge_metadata(metadata)
        if self.remote is not None:
            self.push_changes()
        else:
            self.write_file(directory, META_FILE, project.metadata)
            self.write_file(directory, MANIFEST_FILE, self._tsrc)
            self.write_file(directory, GIT_IGNORE, _gitignore)
            add_and_commit(directory)
