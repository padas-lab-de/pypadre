import os
import unittest

# noinspection PyMethodMayBeStatic
import numpy as np

from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.core.model.code.code_mixin import Function
from pypadre.core.model.dataset.dataset import Transformation
from pypadre.core.model.experiment import Experiment
from pypadre.core.model.pipeline.components.components import CustomSplit
from pypadre.core.model.project import Project
from pypadre.core.util.utils import unpack
from pypadre.pod.importing.dataset.dataset_import import SKLearnLoader
from pypadre.pod.tests.base_test import PadreAppTest

test_numpy_array = np.array([[1.0, "A", 2],
                             [2.0, "B", 2],
                             [3.0, "A", 3],
                             [3.0, "C", 4]])


def create_test_pipeline_SVC():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('SVC', SVC(probability=True))]
    return Pipeline(estimators)


def create_test_pipeline_SVR():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVR
    # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    estimators = [('SVR', SVR())]
    return Pipeline(estimators)

def create_sklearn_test_pipeline(*, estimators, **kwargs):
    def sklearn_pipeline():
        from sklearn.pipeline import Pipeline
        return Pipeline(estimators)

    return SKLearnPipeline(pipeline_fn=sklearn_pipeline, **kwargs)


def create_test_pipeline_multiple_estimators():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.decomposition.pca import PCA
    # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    estimators = [('PCA', PCA()), ('SVC', SVC(probability=True))]
    return Pipeline(estimators)


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def find_subdirectories(path):
    return [o for o in os.listdir(path) if os.path.isdir(os.path.join(path,o))]


class TestSKLearnPipeline(PadreAppTest):

    def __init__(self, *args, **kwargs):
        super(TestSKLearnPipeline, self).__init__(*args, **kwargs)

    def setUp(self):
        self.project = Project(name='Test Project', description='Some description')

    def test_default_sklearn_pipeline(self):

        project = Project(name='Test Project 2',
                          description='Testing the functionalities of project backend',
                          creator=Function(fn=self.test_default_sklearn_pipeline))

        pipeline = SKLearnPipeline(pipeline_fn=create_test_pipeline_SVC)

        loader = SKLearnLoader()
        iris = loader.load("sklearn", utility="load_iris")
        experiment = Experiment(dataset=iris, project=self.project, pipeline=pipeline,
                                reference=self.test_default_sklearn_pipeline)

        experiment.execute()
        print(experiment)
        # TODO asserts and stuff

    def test_custom_split_sklearn_pipeline(self):

        def custom_split(ctx, **kwargs):
            (data,) = unpack(ctx, "data")
            idx = np.arange(data.size[0])
            cutoff = int(len(idx) / 2)
            return idx[:cutoff], idx[cutoff:], None

        pipeline = SKLearnPipeline(splitting=CustomSplit(fn=custom_split), pipeline_fn=create_test_pipeline_SVC)
        iris = SKLearnLoader().load("sklearn", utility="load_iris")
        experiment = Experiment(dataset=iris, project=self.project, pipeline=pipeline,
                                reference=self.test_custom_split_sklearn_pipeline)

        experiment.execute()

        # TODO asserts and stuff

        assert(isinstance(experiment.project, Project))

        assert(experiment.parent is not None)
        assert(experiment.created_at is not None)

        assert(len(experiment.executions) > 0)
        assert(experiment.executions is not None and isinstance(experiment.executions, list))
        assert(experiment.executions[0].parent == experiment)
        assert(isinstance(experiment.pipeline, SKLearnPipeline))

    def test_sklearn_pipeline_with_preprocessing(self):

        def preprocessing(ctx, **kwargs):
            (data,) = unpack(ctx, "data")
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

        pipeline = SKLearnPipeline(preprocessing_fn=Function(fn=preprocessing), pipeline_fn=create_test_pipeline_SVC)

        loader = SKLearnLoader()
        digits = loader.load("sklearn", utility="load_iris")

        experiment = Experiment(name='Test Experiment', project=self.project, dataset=digits, pipeline=pipeline,
                                reference=self.test_sklearn_pipeline_with_preprocessing)

        experiment.execute()

    def test_hyperparameter_search(self):

        pipeline = SKLearnPipeline(pipeline_fn=create_test_pipeline_SVC)

        iris = SKLearnLoader().load("sklearn", utility="load_iris")
        experiment = Experiment(name='Hyperparameter Search', project=self.project,
                                dataset=iris, pipeline=pipeline,
                                reference=self.test_hyperparameter_search)

        params_svc = {'C': [0.5, 1.0, 1.5],
                      'poly_degree': [1, 2, 3],
                      'tolerance': [1, 3]}
        params_dict = {'SVC': params_svc}
        param = {'SKLearnEstimator': params_dict}
        experiment.execute(parameters=param)
        print(experiment)

    def test_full_stack(self):
        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.binding.metrics import sklearn_metrics
        print(sklearn_metrics)
        # TODO plugin system

        self.app.datasets.load_defaults()
        project = Project(name='Test Project 2',
                          description='Testing the functionalities of project backend',
                          creator=Function(fn=self.test_full_stack))

        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

        id = '_iris_dataset'
        dataset = self.app.datasets.list({'name': id})

        experiment = Experiment(name='Test Experiment', description='Test Experiment',
                                dataset=dataset.pop(), project=project,
                                pipeline=SKLearnPipeline(pipeline_fn=create_test_pipeline),
                                reference=self.test_full_stack)
        experiment.execute()
        assert(experiment.executions is not None)
        computations = self.app.computations.list()
        assert(isinstance(computations, list))
        assert(len(computations) > 0)
        experiments = self.app.experiments.list()
        assert(isinstance(experiments, list))
        assert(len(experiments)>0)

    def test_all_functionalities_regression(self):

        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.binding.metrics import sklearn_metrics
        print(sklearn_metrics)
        # TODO plugin system

        self.app.datasets.load_defaults()
        project = Project(name='Test Project Regression',
                          description='Testing the functionalities of project',
                          creator=Function(fn=self.test_all_functionalities_regression))

        _id = '_diabetes_dataset'

        dataset = self.app.datasets.list({'name': _id})

        experiment = Experiment(name='Test Experiment', description='Test Experiment',
                                dataset=dataset.pop(), project=project,
                                pipeline=SKLearnPipeline(pipeline_fn=create_test_pipeline_SVR),
                                reference=self.test_all_functionalities_regression)
        parameter_dict = {'SVR': {'C': [0.1, 0.2]}}
        experiment.execute(parameters={'SKLearnEvaluator': {'write_results': True},
                                       'SKLearnEstimator': {'parameters': parameter_dict}
                                       })

        files_found = find('results.bin',
                           os.path.expanduser('~/.pypadre-test/projects/Test Project Regression/'
                                                             'experiments/Test Experiment/executions'))
        assert (files_found is not None)

        files_found = find('initial_hyperparameters.json',
                           os.path.expanduser('~/.pypadre-test/projects/Test Project Regression/'
                                                                              'experiments/Test Experiment/executions'))
        assert (files_found is not None)

        files_found = find('parameters.json',
                           os.path.expanduser('~/.pypadre-test/projects/Test Project Regression/'
                                                                 'experiments/Test Experiment/executions'))
        assert (files_found is not None)


    def test_custom_split_pipeline(self):

        def custom_split(ctx, **kwargs):
            (data,) = unpack(ctx, "data")
            idx = np.arange(data.size[0])
            cutoff = int(len(idx) / 2)
            return idx[:cutoff], idx[cutoff:], None

        from sklearn.svm import SVC
        from sklearn.decomposition import PCA
        pipeline = create_sklearn_test_pipeline(estimators=[('PCA', PCA()),('SVC', SVC(probability=True))],
                                                splitting=CustomSplit(fn=custom_split))

        self.app.datasets.load_defaults()
        # TODO investigate race condition? dataset seems to be sometimes null in the dataset
        project = self.create_project(name='Test Project Custom Split', description='Testing custom splits',
                                      store_code=True, creator_name="f_" + os.path.basename(__file__), creator_code=__file__)
        dataset = self.app.datasets.list({'name': '_iris_dataset'})
        experiment = self.create_experiment(name='Test Experiment Custom Split', description='Testing custom splits',
                                            dataset=dataset.pop(), project=project, pipeline=pipeline, store_code=True,
                                            creator_name="f_" + os.path.basename(__file__), creator_code=__file__,
                                            creator=self.test_custom_split_pipeline)

        experiment.execute()

        self.app.computations.list()
        experiments = self.app.experiments.list()

    def test_dumping_intermediate_results(self):

        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.binding.metrics import sklearn_metrics
        print(sklearn_metrics)
        # TODO plugin system

        self.app.datasets.load_defaults()
        project = Project(name='Test Project 2',
                          description='Testing the functionalities of project backend',
                          creator=Function(fn=self.test_full_stack))

        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

        id = '_iris_dataset'
        dataset = self.app.datasets.list({'name': id})

        experiment = Experiment(name='Test Experiment', description='Test Experiment',
                                dataset=dataset.pop(), project=project,
                                pipeline=SKLearnPipeline(pipeline_fn=create_test_pipeline),
                                reference=self.test_full_stack)

        experiment.execute(parameters={'SKLearnEvaluator': {'write_results': True}})

        files_found = find('results.bin', os.path.expanduser('~/.pypadre-test/projects/Test Project 2/'
                                                             'experiments/Test Experiment/executions'))
        assert(files_found is not None)


    def test_all_functionalities_classification(self):

        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.binding.metrics import sklearn_metrics
        print(sklearn_metrics)
        # TODO plugin system

        self.app.datasets.load_defaults()
        project = Project(name='Test Project 2',
                          description='Testing the functionalities of project backend',
                          creator=Function(fn=self.test_full_stack))

        _id = '_iris_dataset'
        dataset = self.app.datasets.list({'name': _id})

        experiment = Experiment(name='Test Experiment', description='Test Experiment',
                                dataset=dataset.pop(), project=project,
                                pipeline=SKLearnPipeline(pipeline_fn=create_test_pipeline_multiple_estimators),
                                reference=self.test_full_stack)
        parameter_dict = {'SVC': {'C':[0.1,0.2]}, 'PCA': {'n_components':[1, 2, 3]}}
        experiment.execute(parameters={'SKLearnEvaluator': {'write_results': True},
                                       'SKLearnEstimator': {'parameters': parameter_dict}
                                       })

        files_found = find('results.bin', os.path.expanduser('~/.pypadre-test/projects/Test Project 2/'
                                                             'experiments/Test Experiment/executions'))
        assert (files_found is not None)

        files_found = find('initial_hyperparameters.json', os.path.expanduser('~/.pypadre-test/projects/Test Project 2/'
                                                             'experiments/Test Experiment/executions'))
        assert (files_found is not None)

        files_found = find('parameters.json', os.path.expanduser('~/.pypadre-test/projects/Test Project 2/'
                                                                         'experiments/Test Experiment/executions'))
        assert (files_found is not None)

    def test_multiple_experiments_one_project(self):
        project = Project(name='Test Project Multiple Experiments',
                          description='Testing the functionalities of project backend',
                          creator=Function(fn=self.test_multiple_experiments_one_project))

        self.app.datasets.load_defaults()

        experiment_name1 = 'Test Experiment1'
        experiment_name2 = 'Test Experiment2'

        _id = '_iris_dataset'
        dataset = self.app.datasets.list({'name': _id})

        experiment = Experiment(name=experiment_name1, description='Test Experiment',
                                dataset=dataset.pop(), project=project,
                                pipeline=SKLearnPipeline(pipeline_fn=create_test_pipeline_multiple_estimators),
                                reference=self.test_full_stack)
        parameter_dict = {'SVC': {'C': [0.1, 0.2]}, 'PCA': {'n_components': [1, 2, 3]}}
        experiment.execute(parameters={'SKLearnEvaluator': {'write_results': True},
                                       'SKLearnEstimator': {'parameters': parameter_dict}
                                       })

        dataset = self.app.datasets.list({'name': _id})
        experiment = Experiment(name=experiment_name2, description='Test Experiment',
                                dataset=dataset.pop(), project=project,
                                pipeline=SKLearnPipeline(pipeline_fn=create_test_pipeline_multiple_estimators),
                                reference=self.test_full_stack)
        parameter_dict = {'SVC': {'C': [0.1, 0.2]}, 'PCA': {'n_components': [1, 2, 3]}}
        experiment.execute(parameters={'SKLearnEvaluator': {'write_results': True},
                                       'SKLearnEstimator': {'parameters': parameter_dict}
                                       })

        files_found = find_subdirectories(os.path.expanduser('~/.pypadre-test/projects/Test Project Multiple Experiments/experiments/'))
        assert (experiment_name1 in files_found and experiment_name2 in files_found)

    def test_multiple_projects(self):

        project_name1 = 'Test Project 1 Multiple Projects'
        project_name2 = 'Test Project 2 Multiple Projects'
        project1 = Project(name=project_name1,
                           description='Testing the functionalities of project backend',
                           creator=Function(fn=self.test_multiple_experiments_one_project))

        project2 = Project(name=project_name2,
                           description='Testing the functionalities of project backend',
                           creator=Function(fn=self.test_multiple_experiments_one_project))

        self.app.datasets.load_defaults()

        experiment_name1 = 'Test Experiment1'
        experiment_name2 = 'Test Experiment2'

        _id = '_iris_dataset'
        dataset = self.app.datasets.list({'name': _id})

        experiment = Experiment(name=experiment_name1, description='Test Experiment',
                                dataset=dataset.pop(), project=project1,
                                pipeline=SKLearnPipeline(pipeline_fn=create_test_pipeline_multiple_estimators),
                                reference=self.test_full_stack)
        parameter_dict = {'SVC': {'C': [0.1, 0.2]}, 'PCA': {'n_components': [1, 2, 3]}}
        experiment.execute(parameters={'SKLearnEvaluator': {'write_results': True},
                                       'SKLearnEstimator': {'parameters': parameter_dict}
                                       })

        dataset = self.app.datasets.list({'name': _id})
        experiment = Experiment(name=experiment_name2, description='Test Experiment',
                                dataset=dataset.pop(), project=project2,
                                pipeline=SKLearnPipeline(pipeline_fn=create_test_pipeline_multiple_estimators),
                                reference=self.test_full_stack)
        parameter_dict = {'SVC': {'C': [0.1, 0.2]}, 'PCA': {'n_components': [1, 2, 3]}}
        experiment.execute(parameters={'SKLearnEvaluator': {'write_results': True},
                                       'SKLearnEstimator': {'parameters': parameter_dict}
                                       })

        files_found = find_subdirectories(
            os.path.expanduser('~/.pypadre-test/projects/'))
        assert (project_name1 in files_found and project_name2 in files_found)


if __name__ == '__main__':
    unittest.main()
