import os
import re

from pypadre.pod.backend.interfaces.backend.generic.i_base_file_backend import File
from pypadre.pod.backend.interfaces.backend.i_project_backend import IProjectBackend
from pypadre.pod.backend.local.file.project.experiment.experiment_file_backend import PadreExperimentFileBackend
from pypadre.pod.backend.serialiser import JSonSerializer
from pypadre.core.model.project import Project


class PadreProjectFileBackend(IProjectBackend):

    @staticmethod
    def _placeholder():
        return '{PROJECT_ID}'

    @staticmethod
    def _get_parent_of(obj: Project):
        # Projects have no parents
        return None

    NAME = 'projects'

    def __init__(self, parent):

        super().__init__(parent=parent, name=self.NAME)
        # self.root_dir = os.path.join(self._parent.root_dir, self.NAME, self.PLACEHOLDER)
        self.root_dir = os.path.join(self._parent.root_dir, self.NAME)
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        self._experiment = PadreExperimentFileBackend(self)

    META_FILE = File("metadata.json", JSonSerializer)

    @property
    def experiment(self):
        return self._experiment

    def to_folder_name(self, obj):
        # TODO only name for folder okay? (maybe a uuid, a digest of a config or similar?)
        return obj.name

    def get_by_name(self, name):
        """
        Shortcut because we know name is the folder name. We don't have to search in metadata.json
        :param name: Name of the dataset
        :return:
        """
        return self.list({'folder': re.escape(name)})

    def get_by_dir(self, directory):
        metadata = self.get_file(directory, self.META_FILE)
        if metadata is None:
            # TODO write code
            raise ValueError()
        return Project(**metadata)

    def put(self, project, **kwargs):

        directory = self.to_directory(project)

        # Create the directory with flags, allow_overwrite False and append_data True
        super().put(project, False, True)

        # Create a repo for the project
        # Check if the folder exists, if the folder exists the repo will already be created, else create the repo
        self._create_repo(path=directory, bare=False)

        # Write metadata of the project
        self.write_file(directory, self.META_FILE, project.metadata)
