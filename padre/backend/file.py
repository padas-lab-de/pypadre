"""
  Module handling different repositories. A repository manages datasets and allows to load / store datasets.

  Currently we distinguish between a FileRepository and a HTTPRepository.
  In addition, the module defines serialiser for the individual binary data sets
"""

import os
import re
import shutil

from padre.backend.serialiser import JSonSerializer, PickleSerializer
from padre.datasets import Dataset, Attribute


def _get_path(root_dir, name):
    # internal get or create path function
    _dir = os.path.join(root_dir, name)
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    return _dir

# TODO this repository is outdated and needs to be adapted to the changes in the dataset class (e.g. difference in targets and data)
class PadreFileRepository(object):
    """
    repository as File Directory with the following format

    ```
    root_dir
       |-<dataset name>
               |-- data.bin
               |-- target.bin
               |-- metadata.json
       |-<dataset2 name>
       ...

    """

    def __init__(self, root_dir):
        self.root_dir = _get_path(root_dir, "")
        self._metadata_serializer = JSonSerializer
        self._data_serializer = PickleSerializer

    def list(self, search_id=None, search_metadata=None):
        """
        List all data sets in the repository
        :param search_name: regular expression based search string for the title. Default None
        :param search_metadata: dict with regular expressions per metadata key. Default None
        """
        datasets = os.listdir(self.root_dir)
        if search_id is not None:
            rid = re.compile(search_id)
            datasets = [data for data in datasets if rid.match(data)]

        if search_metadata is not None:
            raise ValueError("metadata search not supported in file repository yet")

        return datasets

    def put(self, id,  dataset):
        _dir = _get_path(self.root_dir, id)
        try:
            with open(os.path.join(_dir, "data.bin"), 'wb') as f:
                f.write(self._data_serializer.serialise(dataset.data))

            if dataset.target is not None:
                with open(os.path.join(_dir, "target.bin"), 'wb') as f:
                    f.write(self._data_serializer.serialise(dataset.target))

            with open(os.path.join(_dir, "metadata.json"), 'w') as f:
                f.write(self._metadata_serializer.serialise(dataset.metadata))

        except Exception as e:
            shutil.rmtree(_dir)
            raise e


    def get(self, id, metadata_only=False):
        """
        Fetches a data set with `name` and returns it (plus some metadata)

        :param id:
        :return: returns the dataset or the metadata if metadata_only is True
        """
        _dir = _get_path(self.root_dir, id)

        with open(os.path.join(_dir, "metadata.json"), 'r') as f:
            metadata = self._metadata_serializer.deserialize(f.read())

        if metadata_only:
            return metadata
        else:
            data = target = None
            with open(os.path.join(_dir, "data.bin"), 'rb') as f:
                data = self._data_serializer.deserialize(f.read())

            t_f = os.path.join(_dir, "target.bin")
            if os.path.exists(t_f) is not None:
                with open(t_f, 'rb') as f:
                    target = self._data_serializer.deserialize(f.read())


            # todo currently, the target is lost.
            dataset = Dataset(id, **metadata)
            dataset.set_data(data)

