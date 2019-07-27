import os
import shutil

from pypadre.backend.interfaces.backend.generic.i_base_file_backend import File
from pypadre.backend.interfaces.backend.i_dataset_backend import IDatasetBackend

from pypadre.backend.serialiser import JSonSerializer, PickleSerializer
from pypadre.core.model.dataset.attribute import Attribute
from pypadre.core.model.dataset.dataset import Dataset


class PadreDatasetFileBackend(IDatasetBackend):

    def __init__(self, parent):
        super().__init__(parent=parent, name="datasets")

    META_FILE = File("metadata.json", JSonSerializer)
    DATA_FILE = File("data.bin", PickleSerializer)

    def put(self, dataset: Dataset, allow_overwrite=True):
        directory = self.get_dir(self.to_folder_name(dataset))

        if os.path.exists(directory) and not allow_overwrite:
            raise ValueError("Dataset %s already exists." +
                             "Overwriting not explicitly allowed. Set allow_overwrite=True".format(dataset.name))
        else:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.mkdir(directory)

        self.write_file(directory, self.META_FILE, dataset.metadata)

    def get_by_dir(self, directory):
        metadata = self.get_file(directory, self.META_FILE)

        attributes = metadata.pop("attributes")
        # print(type(metadata))
        ds = Dataset(id, **metadata)
        # sorted(attributes, key=lambda a: a["index"])
        # assert sum([int(a["index"]) for a in attributes]) == len(attributes) * (
        #    len(attributes) - 1) / 2  # todo check attribute correctness here
        # TODO can this mapping be done in a better way???
        # TODO All of this should be done in the the Dataset constructor
        attributes = [Attribute(a["name"], a["measurementLevel"], a["unit"], a["description"],
                                a["defaultTargetAttribute"], a["context"], a["index"])
                      for a in attributes]
        ds.set_data(None, attributes)
        if self.has_file(directory, self.DATA_FILE):
            ds.set_data(self.get_file(directory, self.DATA_FILE))
        return ds

    def to_folder_name(self, obj):
        """
        Converts the object to a name for the folder (For example the name of a dataset)
        :param obj: dataset passed
        :return:
        """
        return obj.name

    def get_by_name(self, name):
        """
        Shortcut because we know name is the folder name. We don't have to search in metadata.json
        :param name: Name of the dataset
        :return:
        """
        return self.get_by_dir(self.get_dir(name))
