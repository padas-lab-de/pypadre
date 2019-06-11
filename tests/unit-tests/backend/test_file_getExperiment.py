"""
This file contains tests covering backend.file.ExperimentFileRepository.get_experiment
Test all scenarios if expected experiment is loaded from local file system

"""
import os
import shutil

import unittest
import uuid

from pypadre.app import p_app
from pypadre.core import Experiment
from pypadre.ds_import import load_sklearn_toys
from pypadre.backend.serialiser import JSonSerializer


class TestFileGetExperiment(unittest.TestCase):
    """Test file.ExperimentFileRepository.get_experiment with all scenarios
    """

    def setUp(self):
        self.experiment = self.create_experiment()

    def test_get_experiment(self):
        """
        Test ExperimentFileRepository.get_experiment

        Scenarios:
            - get_experiment returns an instance of  <class 'pypadre.experiment.Experiment'>
            - get_experiment associates correct dataset with experiment
            - get_experiment loads expected metadata with the experiment
            - get_experiment loads experiment with expected name
            - get_experiment loads experiment with expected id
            - get_experiment loads expected experiment configuration
        """
        loaded_experiment = p_app.local_backend.experiments.get_experiment(self.experiment.name)
        self.assertIsInstance(loaded_experiment, Experiment, "Not an instance of pypadre.experiment.Experiment")
        self.assertEqual(self.experiment.metadata["dataset_id"],
                         loaded_experiment.metadata["dataset_id"],
                         "Dataset id dont match for both experiments")
        self.assertEqual(self.experiment.name,
                         loaded_experiment.name,
                         "Experiment name not matches for both experiments")
        self.assertEqual(self.experiment.id,
                         loaded_experiment.id,
                         "Experiment id not matches for both experiments")
        self.assertDictEqual(self.experiment.metadata, loaded_experiment.metadata,
                             "Meta data not matches for both experiments")
        self.assertDictEqual(self.experiment.experiment_configuration, loaded_experiment.experiment_configuration,
                             "Experiment configuration not matches for both experiments")

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