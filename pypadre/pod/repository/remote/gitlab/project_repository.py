import os
import re

from pypadre.core.model.project import Project
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IProjectRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.remote.gitlab.repository.gitlab import GitLabRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer

NAME = 'projects'
META_FILE = File("metadata.json", JSonSerializer)


class ProjectGitlabRepository(GitLabRepository, IProjectRepository):

    @staticmethod
    def placeholder():
        return '{PROJECT_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(root_dir=os.path.join(backend.root_dir, NAME),gitlab_url=backend.url,token=backend.token, backend=backend)

    def get_by_dir(self, directory):
        metadata = self.get_file(directory, META_FILE)
        return Project(name=metadata.pop("name"), description=metadata.pop("description"), metadata=metadata)

    def to_folder_name(self, project):
        # TODO only name for folder okay? (maybe a uuid, a digest of a config or similar?)
        return project.name

    def get_by_name(self, name):
        """
        Shortcut because we know name is the folder name. We don't have to search in metadata.json
        :param name: Name of the dataset
        :return:
        """
        return self.list({'folder': re.escape(name)})

    def _put(self, obj, *args, directory: str, remote=None, merge=False, **kwargs):
        project = obj
        self.write_file(directory, META_FILE, project.metadata)
        if remote:
            self.add_and_commit(obj)
            self._remote.push(refspec='{}:{}'.format('master', 'master'))
