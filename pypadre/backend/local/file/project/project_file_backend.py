import os
import shutil

from pypadre.backend.interfaces.backend.generic.i_base_file_backend import File
from pypadre.backend.interfaces.backend.i_project_backend import IProjectBackend
from pypadre.backend.local.file.project.experiment.experiment_file_backend import PadreExperimentFileBackend
from pypadre.backend.serialiser import JSonSerializer


class PadreProjectHTTPBackend(IProjectBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "projects")
        self._experiment = PadreExperimentFileBackend(self)

    META_FILE = File("metadata.json", JSonSerializer)

    @property
    def experiment(self):
        return self._experiment

    def to_folder_name(self, obj):
        # TODO only name for folder okay (maybe a uuid, a digest of a config or similar?)
        return obj.name

    def get_by_name(self, name):
        """
        Shortcut because we know name is the folder name. We don't have to search in metadata.json
        :param name: Name of the dataset
        :return:
        """
        return self.get_by_dir(self.get_dir(name))

    def get_by_dir(self, directory):
        metadata = self.get_file(directory, self.META_FILE)
        # TODO project instance from metadata
        project = {}
        pass
