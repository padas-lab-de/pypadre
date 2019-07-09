import os
import shutil

from pypadre import Dataset
from pypadre.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.backend.interfaces.backend.i_dataset_backend import IDatasetBackend
from pypadre.backend.local.file.interfaces.i_base_file_backend import IBaseBinaryFileBackend
from pypadre.core.model.dataset.attribute import Attribute
from pypadre.util.file_util import dir_list, get_path


class PadreDatasetFileBackend(IDatasetBackend, IBaseBinaryFileBackend, ISearchable):

    def __init__(self, parent):
        super().__init__(parent=parent, name="datasets")

    def list(self, search):
        """
        List all data sets in the repository
        :param search: search object. You can pass key value pairs to search for.
        """

        # todo apply the search filter.
        # todo implement search package
        dirs = dir_list(self.root_dir, "")
        return dirs  # [self.get(dir, metadata_only=True) for dir in dirs]

    def list_id(self, search):
        dirs = self.list(search)
        # todo convert to id list
        return dirs

    def get(self, name):
        """
        Fetches a data set with `name` and returns it (plus some metadata)

        :param name: name of the dataset
        :return: returns the dataset or the metadata if metadata_only is True
        """
        metadata = self.get_meta_file(name)
        attributes = metadata.pop("attributes")
        # print(type(metadata))
        ds = Dataset(id, **metadata)
        # sorted(attributes, key=lambda a: a["index"])
        # assert sum([int(a["index"]) for a in attributes]) == len(attributes) * (
        #    len(attributes) - 1) / 2  # check attribute correctness here
        attributes = [Attribute(a["name"], a["measurementLevel"], a["unit"], a["description"],
                               a["defaultTargetAttribute"], a["context"], a["index"])
                     for a in attributes]
        ds.set_data(None, attributes)
        if self.has_binary_file(name):
            ds.set_data(self.get_binary_file_fn(name))
        return ds

    def put(self, dataset):
        """
        stores the provided dataset into the file backend under the directory `dataset.id`
        (file `data.bin` contains the binary and file `metadata.json` contains the metadata)
        :param dataset: dataset to put.
        :return:
        """
        _dir = self.get_dir(dataset.name)
        try:
            if dataset.has_data():
                with open(os.path.join(_dir, "data.bin"), 'wb') as f:
                    f.write(self._data_serializer.serialise(dataset.data))

            with open(os.path.join(_dir, "metadata.json"), 'w') as f:
                metadata = dict(dataset.metadata)
                metadata["attributes"] = dataset.attributes
                f.write(self._metadata_serializer.serialise(metadata))

        except Exception as e:
            shutil.rmtree(_dir)
            raise e

    def delete(self, uid):
        """
        :param uid: the id of the dataset to delete
        :return:
        """
        if self.has_dir(uid):
            shutil.rmtree(self.get_dir(uid))

    def put_progress(self, obj):
        pass