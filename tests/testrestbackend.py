import shutil
import tempfile
import unittest

import numpy as np

import pypadre.pod.backend.local.file as parep
import pypadre.pod.importing.dataset.ds_import as dsimp
from pypadre.pod.backend.http import PadreHTTPClient

_test_data = {
      "name": "testdata",
      "data": np.random.random((10, 20))
    }

_test_json = {
      "name": "testdata",
      "url": "http://some.url",
      "float": 0.91
    }


# todo requires that the server is running somewhere
# todo check if server is running, otherwise skip test
class TestHTTPBackend(unittest.TestCase):

    client = PadreHTTPClient();

    def test_get(self):
        print(self.client.do_get("datasets"))


class TestRestRepository(unittest.TestCase):

    def test_import_sklearn(self):
        if True:
            return
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
            datasets_restored = repo.list_datasets()
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
    pass
