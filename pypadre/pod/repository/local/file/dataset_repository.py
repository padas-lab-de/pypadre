import json
import os
import re

from pypadre.core.model.dataset.attribute import Attribute
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IDatasetRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.local.file.generic.i_git_repository import IGitRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer, PickleSerializer
from pypadre.pod.util.git_util import add_git_lfs_attribute_file

META_FILE = File("metadata.json", JSonSerializer)
DATA_FILE = File("data.bin", PickleSerializer)


NAME = "datasets"


class DatasetFileRepository(IGitRepository, IDatasetRepository):

    @staticmethod
    def placeholder():
        return '{DATASET_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(root_dir=os.path.join(backend.root_dir, NAME), backend=backend)

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        dataset = obj

        self.write_file(directory, META_FILE, dataset.metadata)
        self.write_file(directory, DATA_FILE, dataset.data(), 'wb')
        add_git_lfs_attribute_file(directory, "*.bin")

    def get_by_dir(self, directory):
        if len(directory) == 0:
            return None

        metadata = self.get_file(directory, META_FILE)

        attributes = [Attribute(**a) for a in metadata.get("attributes", {})]
        metadata["attributes"] = attributes

        ds = Dataset(metadata=metadata)

        if self.has_file(os.path.join(self.root_dir, directory), DATA_FILE):
            # TODO: Implement lazy loading of the dataset
            def _load_data():
                return self.get_file(os.path.join(self.root_dir, directory), DATA_FILE)
            ds.add_proxy_loader(_load_data)
        return ds

    def to_folder_name(self, dataset):
        """
        Converts the object to a name for the folder (For example the name of a dataset)
        :param obj: dataset passed
        :return:
        """
        return dataset.name

    def list(self, search, offset=0, size=100):
        if hasattr(search, "name"):
            # Shortcut because we know name is the folder name. We don't have to search in metadata.json
            name = search.pop("name")
            search['folder'] = re.escape(name)
        return super().list(search, offset, size)

    def put_visualization(self, visualization, *args, **kwargs):
        """
        Put visualization schema in the local file system.

        :param visualization: Visualization schema as string
        :param args:
        :param kwargs: Expected file_name, base_path
        :return:
        """
        visualization = json.loads(visualization)
        file_name = kwargs.pop("file_name", None)
        if file_name is None:
            file_name = "visualization.json"
        self.write_file(kwargs.pop("base_path"), File(file_name, JSonSerializer), visualization)
