import os

from pypadre import Dataset
from pypadre.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.backend.interfaces.backend.i_dataset_backend import IDatasetBackend
from pypadre.backend.local.file.interfaces.i_searchable_file import ISearchableFile
from pypadre.util.file_util import dir_list, get_path


class PadreDatasetFileBackend(IDatasetBackend, ISearchableFile, ISearchable):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "datasets")

    def list(self, search):
        """
        List all data sets in the repository
        :param search: search object. You can pass key value pairs to search for.
        """

        # todo apply the search filter.
        # todo implement search package
        dirs = dir_list(self.root_dir, "")

        Dataset()

        return dirs  # [self.get(dir, metadata_only=True) for dir in dirs]

    def list_id(self, search):
        dirs = self.list(search)
        # todo convert to id list
        return dirs

    def get(self, uid):
        """
                Fetches a data set with `name` and returns it (plus some metadata)

                :param id:
                :return: returns the dataset or the metadata if metadata_only is True
                """
        _dir = _get_path(self.root_dir, id)

        with open(os.path.join(_dir, "metadata.json"), 'r') as f:
            metadata = self._metadata_serializer.deserialize(f.read())
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
        if os.path.exists(os.path.join(_dir, "data.bin")):
            def __load_data():
                with open(os.path.join(_dir, "data.bin"), 'rb') as f:
                    data = self._data_serializer.deserialize(f.read())
                return data, attributes

            ds.set_data(__load_data)
        return ds

    def put(self, obj):
        _dir = _get_path(self.root_dir, str(dataset.name))
        pass

    def delete(self, uid):
        pass

    def put_progress(self, obj):
        pass

    def _get_dir(obj, name):
        return get_path(obj.root_dir, str(name))