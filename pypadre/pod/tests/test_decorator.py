import unittest

import numpy as np
from sklearn.datasets import load_iris

from pypadre.pod.tests.base_test import PadreAppTest


class AppLocalBackends(PadreAppTest):

    def test_workflow(self):

        self.app.datasets.load_defaults()
        id = '_iris_dataset'
        dataset = self.app.datasets.list({'name': id})

        @self.app.workflow(dataset=dataset.pop(), project_name="A", project_description="Ad", experiment_name="B", experiment_description="Bd")
        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

    def test_work_flow_dataset(self):

        @self.app.dataset(name="iris")
        def get_dataset():
            data = load_iris().data
            target = load_iris().target.reshape(-1, 1)
            return np.append(data, target, axis=1)

        @self.app.workflow(dataset="iris", project_name="A", project_description="Ad", experiment_name="B", experiment_description="Bd")
        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)


if __name__ == '__main__':
    unittest.main()
