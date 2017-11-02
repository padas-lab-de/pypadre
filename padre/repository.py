"""
  Module handling different repositories. Currently we distinguish between a FileRepository and a HTTPRepository.

  In addition, the module defines serialiser for the individual binary data sets
"""

import os
import re
import pickle
import json
import shutil
import msgpack_numpy as mn
from .datasets import new_dataset

def _get_path(root_dir, name):
    # internal get or create path function
    _dir = os.path.join(root_dir, name)
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    return _dir

# TODO : we should support faster / more sophisticated / cross-plattform serialisation. for example using pyarrow
# TODO: write PickleSerialiser Test
class PickleSerializer(object):
    """
    Serialiser using pythons pickle.
    """
    def serialise(obj):
        """
        serializes the object and returns a byte object
        :param obj: object to serialise
        :return: byte object (TODO: Specify more precise)
        """
        return pickle.dumps(obj)

    def deserialize(buffer):
        """
        Deserialize a object
        :param buffer:
        :return:
        """
        return pickle.loads(buffer)


class JSonSerializer:

    def serialise(obj):
        return json.dumps(obj)

    def deserialize(buffer):
        return json.loads(buffer)

class MsgPack:

    def serialise(obj):
        return mn.dumps(obj)

    def deserialize(buffer):
        return mn.loads(buffer)


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

    def list(self, search_title=None, search_metadata=None):
        """
        List all data sets in the repository
        :param search_title: regular expression based search string for the title. Default None
        :param search_metadata: dict with regular expressions per metadata key. Default None
        """
        datasets = os.listdir(self.root_dir)
        if search_title is not None:
            rtitle = re.compile(search_title)
            datasets = [data for data in datasets if rtitle.match(data)]

        if search_metadata is not None:
            raise ValueError("metadata search not supported in file repository yet")

        return datasets

    def put(self, name,  dataset):
        _dir = _get_path(self.root_dir, name)
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


    def get(self, name, metadata_only=False):
        """
        Fetches a data set with `name` and returns it (plus some metadata)

        :param name:
        :return: returns the dataset or the metadata if metadata_only is True
        """
        _dir = _get_path(self.root_dir, name)

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


            # TODO: let every class register a type (i.e. via a static variable) and check the corresponding class dynamically
            return new_dataset(name, metadata, data, target)


class PadreHttpRepositoryClient(object):

    def __init__(self):
        pass
