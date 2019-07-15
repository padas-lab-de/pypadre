from pypadre import Dataset
from pypadre.backend.interfaces.backend.i_dataset_backend import IDatasetBackend
from pypadre.backend.local.file.interfaces.i_base_binary_file_backend import IBaseBinaryFileBackend
from pypadre.core.model.dataset.attribute import Attribute


class PadreDatasetFileBackend(IDatasetBackend, IBaseBinaryFileBackend):

    def __init__(self, parent):
        super().__init__(parent=parent, name="datasets")

    def get_by_dir(self, directory):
        metadata = self.get_meta_file(directory)
        attributes = metadata.pop("attributes")
        # print(type(metadata))
        ds = Dataset(id, **metadata)
        # sorted(attributes, key=lambda a: a["index"])
        # assert sum([int(a["index"]) for a in attributes]) == len(attributes) * (
        #    len(attributes) - 1) / 2  # todo check attribute correctness here

        # TODO can this mapping be done in a better way???
        attributes = [Attribute(a["name"], a["measurementLevel"], a["unit"], a["description"],
                                a["defaultTargetAttribute"], a["context"], a["index"])
                      for a in attributes]
        ds.set_data(None, attributes)
        if self.has_binary_file(directory):
            ds.set_data(self.get_binary_file_fn(directory))
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
