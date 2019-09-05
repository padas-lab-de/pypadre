import os
import random
import shutil
import tempfile
import unittest

import numpy as np

import pypadre.core.backend.local.file as parep
import pypadre.ds_import as dsimp


# todo renew file repository
_test_data = {
      "name": "testdata",
      "data": np.random.random((10, 20))
    }

_test_json = {
      "name": "testdata",
      "url": "http://some.url",
      "float": 0.91
    }

class TestSerailising(unittest.TestCase):

    def test_pickle(self):
        buf = parep.PickleSerializer.serialise(_test_data)
        restored = parep.PickleSerializer.deserialize(buf)
        assert restored["name"] == _test_data["name"]
        assert np.array_equal(restored["data"], _test_data["data"])

    def test_json(self):
        buf = parep.JSonSerializer.serialise(_test_json)
        restored = parep.JSonSerializer.deserialize(buf)
        for k in _test_json.keys():
            assert restored[k] == _test_json[k]

    def test_msgpack(self):
        buf = parep.MsgPack.serialise(_test_data)
        restored = parep.MsgPack.deserialize(buf)
        assert restored["name"] == _test_data["name"]
        assert np.array_equal(restored["data"], _test_data["data"])


class TestFileRepository(unittest.TestCase):

    def test_file_create(self):
        tmp_dir = tempfile.gettempdir()
        file_name = "_".join([str(i) for i in random.sample(range(10000000000), 4)])
        joint_dir = os.path.join(tmp_dir, file_name)
        if os.path.exists(joint_dir):
            os.rmdir(joint_dir)
        parep.DatasetFileRepository(joint_dir)  # directory should be created here
        assert os.path.exists(joint_dir)
        os.rmdir(joint_dir)

        joint_dir = tempfile.mkdtemp()
        parep.DatasetFileRepository(joint_dir)
        os.rmdir(joint_dir)

    def test_import_sklearn(self):
        # create repository
        _dir = tempfile.mkdtemp()
        repo = parep.DatasetFileRepository(_dir)
        try:
            datasets = []
            for i in dsimp.load_sklearn_toys():
                datasets.append(i.name)
                repo.put_dataset(i.name, i)
                print(i)
            # Read list of datasets
            datasets_restored = repo.list()
            assert set(datasets) == set(datasets_restored), "Names not equal. "
            # Load datasets
            for n in datasets_restored:
                metadata = repo.get_dataset(n, True)
                assert metadata["description"] is not None
                dataset = repo.get_dataset(n)
                assert dataset.data is not None
                assert type(dataset.data) == np.ndarray
                assert dataset.name == n
        finally:
            shutil.rmtree(_dir)


if __name__ == '__main__':
    unittest.main()
