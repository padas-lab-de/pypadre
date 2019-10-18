import unittest

# noinspection PyMethodMayBeStatic
import numpy as np

from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.core.model.code.function import Function
from pypadre.core.model.dataset.dataset import Transformation
from pypadre.core.model.experiment import Experiment
from pypadre.core.model.project import Project
from pypadre.pod.importing.dataset.dataset_import import SKLearnLoader
from pypadre.pod.tests.base_test import PadreAppTest

test_numpy_array = np.array([[1.0, "A", 2],
                             [2.0, "B", 2],
                             [3.0, "A", 3],
                             [3.0, "C", 4]])


def create_test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('SVC', SVC(probability=True))]
    return Pipeline(estimators)


class TestSKLearnPipeline(PadreAppTest):

    def __init__(self, *args, **kwargs):
        super(TestSKLearnPipeline, self).__init__(*args, **kwargs)
        self.project = Project(name='Test Project', description='Some description')

    def test_default_sklearn_pipeline(self):
        # TODO clean up experiment creator
        pipeline = SKLearnPipeline(pipeline_fn=create_test_pipeline)

        loader = SKLearnLoader()
        iris = loader.load("sklearn", utility="load_iris")
        experiment = Experiment(dataset=iris, project=self.project, pipeline=pipeline)

        experiment.execute()
        print(experiment)
        # TODO asserts and stuff

    def test_default_sklearn_pipeline_grid_search(self):
        pass
        # TODO test grid search

    def test_custom_split_sklearn_pipeline(self):

        def custom_split(idx):
            cutoff = int(len(idx) / 2)
            return idx[:cutoff], idx[cutoff:], None

        # TODO please implement custom split function for this example
        pipeline = SKLearnPipeline(splitting=Function(fn=custom_split), pipeline_fn=create_test_pipeline)

        loader = SKLearnLoader()
        iris = loader.load("sklearn", utility="load_iris")
        experiment = Experiment(dataset=iris, project=self.project, pipeline=pipeline)

        experiment.execute()

        # TODO asserts and stuff

        assert(isinstance(experiment.project, Project))

        assert(experiment.parent is not None)
        assert(experiment.createdAt is not None)

        assert(len(experiment.executions) > 0)
        assert(experiment.executions is not None and isinstance(experiment.executions, list))
        assert(experiment.executions[0].parent == experiment)
        assert(isinstance(experiment.pipeline, SKLearnPipeline))

    def test_sklearn_pipeline_with_preprocessing(self):

        def preprocessing(*, data, **kwargs):
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            PCA_ = PCA()
            scaler = StandardScaler()
            _data = Transformation(name="transformed_%s"%data.name, dataset=data)
            features = scaler.fit_transform(data.features())
            new_features = PCA_.fit_transform(features)
            targets = data.targets()
            new_data = np.hstack((new_features, targets))
            _data.set_data(new_data, attributes=data.attributes)
            return _data

        pipeline = SKLearnPipeline(preprocessing_fn=preprocessing, pipeline_fn=create_test_pipeline)

        loader = SKLearnLoader()
        digits = loader.load("sklearn", utility="load_iris")

        experiment = Experiment(name='Test Experiment', project=self.project, dataset=digits, pipeline=pipeline)

        experiment.execute()

    def test_hyperparameter_search(self):

        pipeline = SKLearnPipeline(pipeline_fn=create_test_pipeline)

        loader = SKLearnLoader()
        iris = loader.load("sklearn", utility="load_iris")
        experiment = Experiment(name='Hyperparameter Search', project=self.project,
                                dataset=iris, pipeline=pipeline)

        params_svc = {'C': [0.5, 1.0, 1.5],
                      'poly_degree': [1, 2, 3],
                      'tolerance': [1, 3]}
        params_dict = {'SVC': params_svc}
        param = {'SKLearnEstimator': params_dict}
        experiment.execute(parameters=param)
        print(experiment)


if __name__ == '__main__':
    unittest.main()
