import unittest

import numpy as np

from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.core.model.code.function import Function
from pypadre.core.model.pipeline.components import CustomSplit
from pypadre.core.util.utils import unpack
from pypadre.pod.tests.base_test import PadreAppTest
from pypadre.pod.tests.util.util import create_sklearn_test_pipeline


class AppLocalBackends(PadreAppTest):

    def test_dataset(self):
        # TODO test putting, fetching, searching, folder/git structure, deletion, git functionality?

        # Puts all the datasets
        self.app.datasets.load_defaults()

        # Gets a dataset by name
        id = '_boston_dataset'
        datasets = self.app.datasets.list({'name': id})
        for dataset in datasets:
            assert id in dataset.name

    def test_code(self):
        def foo(ctx):
            return "foo"

        foo_code = self.app.code.create(clz=Function, fn=foo)
        self.app.code.put(foo_code)
        code_list = self.app.code.list()
        loaded_code = code_list.pop()

        out = loaded_code.call()
        assert out is "foo"

    def test_project(self):
        from pypadre.core.model.project import Project
        project = self.create_project(name='Test Project', description='Testing the functionalities of project backend')
        self.app.projects.put(project)

        project2 = self.app.projects.create(name='Test Project2', description='Testing Project')
        assert (isinstance(project, Project))

        name = 'Test Project'
        projects = self.app.projects.list({'name': name})
        for project in projects:
            assert name in project.name

    def test_experiment(self):
        from pypadre.core.model.experiment import Experiment
        project = self.create_project(name='Test Project 2',
                                      description='Testing the functionalities of project backend')
        self.app.projects.put(project)

        self.app.datasets.load_defaults()
        dataset = self.app.datasets.list({'name': '_boston_dataset'})

        from sklearn.svm import SVC
        experiment = self.create_experiment(dataset=dataset.pop(), project=project,
                                            pipeline=create_sklearn_test_pipeline(
                                                estimators=[('SVC', SVC(probability=True))]))

        self.app.experiments.put(experiment)
        name = 'Test Experiment SVM'
        experiments = self.app.experiments.list({'name': name})
        assert (isinstance(experiments, list))
        for experiment in experiments:
            assert (isinstance(experiment, Experiment))
            assert name in experiment.name

    def test_execution(self):

        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.core.model.execution import Execution

        self.app.datasets.load_defaults()
        id = '_boston_dataset'
        dataset = self.app.datasets.list({'name': id})

        project = Project(name='Test Project 2', description='Testing the functionalities of project backend')
        self.app.projects.put(project)

        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

        experiment = Experiment(name="Test Experiment SVM",
                                description="Testing Support Vector Machines via SKLearn Pipeline",
                                dataset=dataset[0],
                                workflow=create_test_pipeline(), keep_splits=True, strategy="random", project=project)
        self.app.experiments.put(experiment)

        codehash = 'abdauoasg45qyh34t'
        execution = Execution(experiment, codehash=codehash, command=None, append_runs=True, parameters=None,
                              preparameters=None, single_run=True,
                              single_transformation=True)
        self.app.executions.patch(execution)

        executions = self.app.executions.list({'hash': codehash})
        for execution_ in executions:
            assert codehash in execution_.hash
        if len(executions) > 0:
            execution = self.app.executions.get(executions.__iter__().__next__().id)
            assert execution[0].name == executions[0].name
        # FIXME Christofer put asserts here

    def test_run(self):
        """
        project_backend: ProjectFileRepository = self.backend.project
        experiment_backend: ExperimentFileRepository = project_backend.experiment
        execution_backend: ExecutionFileRepository = experiment_backend.execution
        run_backend: RunFileRepository = execution_backend.run
        """

        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.core.model.execution import Execution
        from pypadre.core.model.computation.run import Run

        # TODO clean up testing (tests should be independent and only test the respective functionality)
        self.app.datasets.load_defaults()
        id = '_boston_dataset'
        dataset = self.app.datasets.list({'name': id})

        project = Project(name='Test Project 2', description='Testing the functionalities of project backend')
        self.app.projects.put(project)

        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

        experiment = Experiment(name="Test Experiment SVM",
                                description="Testing Support Vector Machines via SKLearn Pipeline",
                                dataset=dataset[0],
                                workflow=create_test_pipeline, keep_splits=True, strategy="random", project=project)
        self.app.experiments.put(experiment)

        codehash = 'abdauoasg45qyh34t'
        execution = Execution(experiment, codehash=codehash, command=None, append_runs=True, parameters=None,
                              preparameters=None, single_run=True,
                              single_transformation=True)
        self.app.executions.put(execution)

        executions = self.app.executions.list({'hash': codehash})
        if len(executions) > 0:
            run = Run(execution=executions[0], workflow=execution.experiment.pipeline, keep_splits=True)
            self.app.runs.put(run)

        else:
            raise ValueError('Execution not listed for the same code has')

        runs = self.app.runs.list(None)

        assert (isinstance(runs, list))
        for run in runs:
            assert (isinstance(run, Run))

    def test_split(self):

        self.app.datasets.load_defaults()
        project = self.app.projects.service.create(name='Test Project 2',
                                                   description='Testing the functionalities of project backend')

        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

        id = '_iris_dataset'
        dataset = self.app.datasets.list({'name': id})

        experiment = self.create_experiment(name="Test Experiment SVM",
                                            description="Testing Support Vector Machines via SKLearn Pipeline",
                                            dataset=dataset[0],
                                            workflow=create_test_pipeline(), keep_splits=True,
                                            strategy="random", project=project)

        codehash = 'abdauoasg45qyh34t'
        execution = self.create_execution(experiment, codehash=codehash, command=None, append_runs=True,
                                          parameters=None,
                                          preparameters=None, single_run=True,
                                          single_transformation=True)

        run = self.create_run(execution=execution, pipeline=execution.experiment.pipeline, keep_splits=True)
        split = self.create_split(run=run, num=0, train_idx=list(range(1, 1000 + 1)), val_idx=None,
                                  test_idx=list(range(1000, 1100 + 1)), keep_splits=True)
        self.app.splits.put(split)
        # FIXME Christofer put asserts here

    def test_full_stack(self):
        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment

        self.app.datasets.load_defaults()
        project = Project(name='Test Project 2', description='Testing the functionalities of project backend')

        def create_test_pipeline():
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC
            # estimators = [('reduce_dim', PCA()), ('clf', SVC())]
            estimators = [('SVC', SVC(probability=True))]
            return Pipeline(estimators)

        id = '_iris_dataset'
        dataset = self.app.datasets.list({'name': id})

        experiment = Experiment(dataset=dataset.pop(), project=project,
                                pipeline=SKLearnPipeline(pipeline_fn=create_test_pipeline))

        experiment.execute()
        self.app.computations.list()
        experiments = self.app.experiments.list()
        # FIXME Christofer put asserts here

    def test_custom_split_pipeline(self):

        def custom_split(ctx, **kwargs):
            (data,) = unpack(ctx, "data")
            idx = np.arange(data.size[0])
            cutoff = int(len(idx) / 2)
            return idx[:cutoff], idx[cutoff:], None

        from sklearn.svm import SVC
        pipeline = create_sklearn_test_pipeline(estimators=[('SVC', SVC(probability=True))],
                                                splitting=CustomSplit(fn=custom_split))

        self.app.datasets.load_defaults()
        # TODO investigate race condition? dataset seems to be sometimes null in the dataset
        project = self.create_project(name='Test Project Custom Split', description='Testing custom splits',
                                      store_code=True, code=__file__)
        dataset = self.app.datasets.list({'name': '_iris_dataset'})
        experiment = self.create_experiment(name='Test Experiment Custom Split', description='Testing custom splits',
                                            dataset=dataset.pop(), project=project, pipeline=pipeline, store_code=True,
                                            code=__file__)

        experiment.execute()

        self.app.computations.list()
        experiments = self.app.experiments.list()
        # FIXME Christofer put asserts here


if __name__ == '__main__':
    unittest.main()
