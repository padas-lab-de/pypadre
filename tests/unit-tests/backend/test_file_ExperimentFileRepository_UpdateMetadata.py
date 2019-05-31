"""
Test backend.file.ExperimentFileRepository.update_metadata function
"""
import os
import shutil
import uuid
import unittest


from padre.app import pypadre
from padre.backend.serialiser import JSonSerializer
from padre.core import Experiment
from padre.ds_import import load_sklearn_toys


def create_test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('SVC', SVC(probability=True))]
    return Pipeline(estimators)


def split(idx):
    limit = int(.7 * len(idx))
    return idx[0:limit], idx[limit:], None


class TestUpdateMetadata(unittest.TestCase):
    """Test file.ExperimentFileRepository.update_metadata

    All unnecessary function calls and http calls are mocked
    """
    def setUp(self):
        """Initializing http client and other attributes for test.

        All non related function calls and http calls will be mocked for this purpose.
        """
        self._metadata_serializer = JSonSerializer
        self.experiment_name = "Test Experiment metadata update " + str(uuid.uuid4())[0:10]
        ex = Experiment(name=self.experiment_name,
                        description="Testing Support Vector Machines via SKLearn Pipeline",
                        dataset=[i for i in load_sklearn_toys()][2],
                        workflow=create_test_pipeline(), keep_splits=True, strategy="random",
                        function=split)
        ex.execute()
        self.experiment_path = os.path.join(pypadre.local_backend.root_dir,
                                            "experiments",
                                            self.experiment_name.strip() + ".ex")

    def test_update_metadata(self):
        """Test metadata is updated for experiment."""
        url = "http://test.com/api/experiments/ex_id"
        pypadre.local_backend.experiments.update_metadata({"server_url": url}, self.experiment_name)
        with open(os.path.join(self.experiment_path, "metadata.json"), 'r') as f:
            self.experiment_metadata = self._metadata_serializer.deserialize(f.read())
            self.assertEqual(url, self.experiment_metadata["server_url"], "Metadata not updated for experiment")

    def tearDown(self):
        """Remove experiment and dataset from local which is created for test purposes"""
        if os.path.exists(os.path.abspath(self.experiment_path)):
            shutil.rmtree(self.experiment_path)

        dataset_root_path = os.path.join(pypadre.local_backend.root_dir, "datasets")
        list_of_datasets = os.listdir(dataset_root_path)
        for dataset_name in list_of_datasets:
            metadata_path = os.path.join(dataset_root_path, dataset_name, "metadata.json")
            if os.path.exists(os.path.abspath(metadata_path)):
                with open(metadata_path, 'r') as f:
                    metadata = self._metadata_serializer.deserialize(f.read())
                    if metadata["id"] == self.experiment_metadata["dataset_id"]:
                        shutil.rmtree(os.path.join(dataset_root_path, dataset_name))
                        break
