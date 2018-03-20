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
    _dir = os.path.expanduser(os.path.join(root_dir, name))
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    return _dir


class PadreFileBackend(object):
    """
    Delegator class for handling padre objects at the file repository. The following files tructure is used:

    root_dir
      |------datasets\
      |------experiments\
    """

    def __init__(self, root_dir):
        self.root_dir = _get_path(root_dir, "")
        self._dataset_repository = DatasetFileRepository(os.path.join(root_dir, "datasets"))
        self._experiment_repository = None


    def datasets(self):
        return self._dataset_repository


class ExperimentFileRepository:
    """
    repository for handling experiments as File Directory with the following format

    ```
    root_dir
       |-<experiment name>
               |-- metadata.json
               |-- aggregated_scores.json
               |-- runs
                  |-- scores.json
                  |-- events.json
                  |-- split 0
                      |-- model.bin
                      |-- split_idx.bin
                      |-- results.bin
                      |-- log.json
                  |-- split 1
                      .....
       |-<experiment2 name>
       ...

    Note that `events.json` and `scores.json` contain the events / scores of the individual splits.
    So logically they would belong to the splits.
    However, for convenience reasons they are aggregated at the run level.
    """

    def __init__(self, root_dir):
        self.root_dir = _get_path(root_dir, "")
        self._metadata_serializer = JSonSerializer
        self._data_serializer = PickleSerializer

    def list(self, search_id=None, search_metadata=None):
        pass

    def put_experiment(self, experiment):
        pass

    def get_experiment(self, name, load_split=None):
        pass



class DatasetFileRepository(object):
    """
    repository for handling datasets as File Directory with the following format

    ```
    root_dir
       |-<dataset name>
               |-- data.bin
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
        # todo implement search in metadata using some kind of syntax (e.g. jsonpath, grep),
        # then search the metadata files one by one.
        datasets = os.listdir(self.root_dir)
        if search_id is not None:
            rid = re.compile(search_id)
            datasets = [data for data in datasets if rid.match(data)]

        if search_metadata is not None:
            raise NotImplemented()

        return datasets

    def put(self, dataset):
        _dir = _get_path(self.root_dir, dataset.id)
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


    def get(self, id, metadata_only=False):
        """
        Fetches a data set with `name` and returns it (plus some metadata)

        :param id:
        :return: returns the dataset or the metadata if metadata_only is True
        """
        _dir = _get_path(self.root_dir, id)

        with open(os.path.join(_dir, "metadata.json"), 'r') as f:
            metadata = self._metadata_serializer.deserialize(f.read())
            attributes = metadata.pop("attributes")

        ds = Dataset(id, metadata)
        sorted(attributes, key=lambda a: a["index"])
        assert sum([int(a["index"]) for a in attributes]) == len(attributes) * (
            len(attributes) - 1) / 2  # check attribute correctness here
        ds.set_data(None,
                    [Attribute(a["name"], a["measurementLevel"], a["unit"], a["description"],
                               a["defaultTargetAttribute"])
                     for a in attributes])
        if metadata_only:
            return ds
        elif os.path.exists(os.path.join(_dir, "data.bin")):
            data = None
            with open(os.path.join(_dir, "data.bin"), 'rb') as f:
                data = self._data_serializer.deserialize(f.read())
            ds.set_data(data, ds.attributes)

