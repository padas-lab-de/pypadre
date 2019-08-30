import os
import shutil

from pypadre.backend.interfaces.backend.generic.i_base_file_backend import File
from pypadre.backend.interfaces.backend.i_project_backend import IProjectBackend
from pypadre.backend.local.file.project.experiment.experiment_file_backend import PadreExperimentFileBackend
from pypadre.backend.serialiser import JSonSerializer
from pypadre.core.model.project import Project


class PadreProjectFileBackend(IProjectBackend):

    NAME = 'projects'
    PLACEHOLDER = '{PROJECT_ID}'

    def __init__(self, parent):

        super().__init__(parent=parent, name=self.NAME)
        #self.root_dir = os.path.join(self._parent.root_dir, self.NAME, self.PLACEHOLDER)
        self.root_dir = os.path.join(self._parent.root_dir, self.NAME)
        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)

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
        return self.get_by_dir(self.get_dir(name))

    def get_by_dir(self, directory):
        metadata = self.get_file(os.path.join(self.root_dir, directory), self.META_FILE)
        return Project(**metadata)

    def put(self, project):

        directory = self.directory(project)
        # Create the directory with flags, allow_overwrite False and append_data True
        super().put(project, False, True)
        if self.PLACEHOLDER in directory:
            directory = directory.replace(self.PLACEHOLDER, project.name)
        # Create a repo for the project
        # Check if the folder exists, if the folder exists the repo will already be created, else create the repo

        if not os.path.exists(directory):
            self._create_repo(path=directory, bare=False)

        # Write metadata of the project
        self.write_file(directory, self.META_FILE, project.metadata)
