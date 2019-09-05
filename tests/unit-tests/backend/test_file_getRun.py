"""
This file contains tests covering backend.file.ExperimentFileRepository.get_run
Test all scenarios for expected run which is loaded from local file system

"""
import os
import shutil

import unittest
import uuid

from pypadre.app import p_app
from pypadre.core import Experiment, Run
from pypadre.ds_import import load_sklearn_toys
from pypadre.core.backend.serialiser import JSonSerializer


class TestFileGetRun(unittest.TestCase):
    """Test file.ExperimentFileRepository.get_run with all possible scenarios
    """

    def setUp(self):
        self.experiment = self.create_experiment()

    def test_get_run(self):
        """
        Test ExperimentFileRepository.get_run

        Scenarios:
            - get_run returns an instance of  <class 'pypadre.core.run.Run'>
            - get_run loads expected metadata with the run
            - get_run loads run with expected name
            - get_run loads run with expected id
            - loaded run is associated with expected experiment
        """
        actual_run = self.experiment.runs[0]
        run_id = str(actual_run.id)
        loaded_run = p_app.local_backend.experiments.get_run(self.experiment.id, run_id)

        self.assertIsInstance(loaded_run, Run, "Not an instance of pypadre.core.run.Run")
        self.assertEqual(actual_run.name, loaded_run.name, "Run name not matches for both runs")
        self.assertEqual(run_id, loaded_run.id, "Run id not matches for both runs")
        self.assertEqual(actual_run.experiment.id, loaded_run.experiment.id,
                         "Expected experiment id for run not matches")
        self.assertDictEqual(actual_run.metadata, loaded_run.metadata,
                             "Run metadata not matches with expected metadata")

    def tearDown(self) -> None:
        """Delete experiment and dataset which is created for this experiment from local file system"""
        metadata_serializer = JSonSerializer
        experiment_path = os.path.join(p_app.local_backend.root_dir,
                                       "experiments",
                                       self.experiment.name.strip() + ".ex")
        if os.path.exists(os.path.abspath(experiment_path)):
            shutil.rmtree(experiment_path)

        dataset_root_path = os.path.join(p_app.local_backend.root_dir, "datasets")
        list_of_datasets = os.listdir(dataset_root_path)
        for dataset_name in list_of_datasets:
            metadata_path = os.path.join(dataset_root_path, dataset_name, "metadata.json")
            if os.path.exists(os.path.abspath(metadata_path)):
                with open(metadata_path, 'r') as f:
                    metadata = metadata_serializer.deserialize(f.read())
                    if metadata["id"] == self.experiment.metadata["dataset_id"]:
                        shutil.rmtree(os.path.join(dataset_root_path, dataset_name))
                        break

    def create_experiment(self):
        experiment_name = "Test experiment " + str(uuid.uuid4())[0:15]
        ds = [i for i in load_sklearn_toys()][2]
        ex = Experiment(name=experiment_name,
                        description="Testing Support Vector Machines via SKLearn Pipeline",
                        dataset=ds,
                        workflow=self.create_test_pipeline(), keep_splits=True, strategy="random",
                        function=self.split, preprocessing=self.create_preprocessing_pipeline())
        ex.execute()
        return ex

    def create_test_pipeline(self):
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC
        estimators = [('SVC', SVC(probability=True))]
        return Pipeline(estimators)

    def create_preprocessing_pipeline(self):
        from sklearn.pipeline import Pipeline
        from sklearn.decomposition import PCA
        estimators = [('PCA', PCA())]
        return Pipeline(estimators)

    def split(self, idx):
        limit = int(.7 * len(idx))
        return idx[0:limit], idx[limit:], None