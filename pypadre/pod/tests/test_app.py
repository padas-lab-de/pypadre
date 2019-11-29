import unittest

from pypadre.core.model.code.code_mixin import Function
from pypadre.pod.tests.base_test import PadreAppTest
from pypadre.pod.tests.util.util import create_sklearn_test_pipeline


class AppLocalBackends(PadreAppTest):

    def setUp(self):
        self.setup_reference(__file__)

    def test_dataset(self):
        # TODO test putting, fetching, searching, folder/git structure, deletion, git functionality?

        # Puts all the datasets
        self.app.datasets.load_defaults()

        # Gets a dataset by name
        id = '_boston_dataset'
        datasets = self.app.datasets.list({'name': id})
        for dataset in datasets:
            assert id in dataset.name

    # def test_code(self):
    #     def foo(ctx):
    #         return "foo"
    #
    #     foo_code = self.app.code.create(clz=Function, fn=foo, repository_identifier=self.test_reference.repository_identifier)
    #     self.app.code.put(foo_code, store_code=True)
    #     code_list = self.app.code.list()
    #     loaded_code = code_list.pop(-1)
    #
    #     out = loaded_code.call()
    #     assert out is "foo"

    def test_project(self):
        from pypadre.core.model.project import Project
        project = self.create_project(name='Test Project', description='Testing the functionalities of project backend')
        self.app.projects.put(project)

        project = self.app.projects.create(name='Test Project2', description='Testing Project')

        assert (isinstance(project, Project))

        name = 'Test Project'
        projects = self.app.projects.list({'name': name})
        for project in projects:
            assert name in project.name

    def test_experiment(self):
        from pypadre.core.model.experiment import Experiment
        project = self.create_project(name='Test experiment',
                                      description='Testing the functionalities of project backend')

        self.app.datasets.load_defaults()
        dataset = self.app.datasets.list({'name': '_boston_dataset'})

        from sklearn.svm import SVC
        experiment = self.create_experiment(name='Test Experiment SVM', description='Testing experiment using SVM',
                                            dataset=dataset.pop(), project=project,
                                            pipeline=
                                            create_sklearn_test_pipeline(estimators=[('SVC', SVC(probability=True))],
                                                                         reference=self.test_reference))

        self.app.experiments.put(experiment)
        name = 'Test Experiment SVM'
        experiments = self.app.experiments.list({'name': name})
        assert (isinstance(experiments, list))
        for experiment in experiments:
            assert (isinstance(experiment, Experiment))
            assert name in experiment.name

    def test_execution(self):

        from pypadre.core.model.project import Project
        from pypadre.core.model.execution import Execution

        self.app.datasets.load_defaults()
        id = '_boston_dataset'
        dataset = self.app.datasets.list({'name': id})

        project = Project(name='Test execution', description='Testing the functionalities of project backend')
        self.app.projects.put(project)

        from sklearn.svm import SVC
        experiment = self.create_experiment(name='Test Experiment SVM', description='Testing experiment using SVM',
                                            dataset=dataset.pop(), project=project,
                                            pipeline=
                                            create_sklearn_test_pipeline(estimators=[('SVC', SVC(probability=True))],
                                                                         reference=self.test_reference))
        self.app.experiments.put(experiment)

        execution = Execution(experiment, reference=self.test_reference, pipeline=experiment.pipeline)
        self.app.executions.patch(execution)

        from pypadre.core.util.utils import persistent_hash
        import pyhash
        execution_id = str(
            self.test_reference.id) + "-" + str(persistent_hash(experiment.id, algorithm=pyhash.city_64()))
        executions = self.app.executions.list({'id': execution_id})
        assert len(executions) == 1

        assert execution_id in executions[0].id
        assert executions[0].experiment_id == experiment.id

        if len(executions) > 0:
            execution = self.app.executions.get(executions.__iter__().__next__().id)
            assert execution[0].name == executions[0].name

    def test_run(self):
        from pypadre.core.model.project import Project
        from pypadre.core.model.experiment import Experiment
        from pypadre.core.model.execution import Execution
        from pypadre.core.model.computation.run import Run

        self.app.datasets.load_defaults()
        id = '_boston_dataset'
        dataset = self.app.datasets.list({'name': id})

        project = Project(name='Test run', description='Testing the functionalities of project backend')
        self.app.projects.put(project)

        from sklearn.svm import SVC
        experiment = Experiment(name="Test Experiment SVM",
                                description="Testing Support Vector Machines via SKLearn Pipeline",
                                dataset=dataset[0],
                                pipeline=create_sklearn_test_pipeline(estimators=[('SVC', SVC(probability=True))],
                                                                      reference=self.test_reference),
                                keep_splits=True, strategy="random", project=project, reference=self.test_reference)
        self.app.experiments.put(experiment)

        execution = Execution(experiment, reference=self.test_reference, pipeline=experiment.pipeline)
        self.app.executions.put(execution)

        from pypadre.core.util.utils import persistent_hash
        import pyhash
        execution_id = str(
            self.test_reference.id) + "-" + str(persistent_hash(experiment.id, algorithm=pyhash.city_64()))

        executions = self.app.executions.list({'id': execution_id})
        assert len(executions) > 0
        run = Run(execution=executions[0])
        run_id = run.id
        self.app.runs.put(run)

        runs = self.app.runs.list(None)

        assert (isinstance(runs, list))
        for run in runs:
            assert (isinstance(run, Run))

        run_ = self.app.runs.get(run_id)
        assert len(run_) == 1
        assert run == run_[0]

    # def test_split(self):
    #     from pypadre.core.model.split.split import Split
    #
    #     self.app.datasets.load_defaults()
    #     project = self.app.projects.service.create(name='Test Project 2',
    #                                                description='Testing the functionalities of project backend')
    #
    #     id = '_iris_dataset'
    #     dataset = self.app.datasets.list({'name': id})
    #
    #     from sklearn.svm import SVC
    #     experiment = self.create_experiment(name="Test Experiment SVM",
    #                                         description="Testing Support Vector Machines via SKLearn Pipeline",
    #                                         dataset=dataset[0],
    #                                         pipeline=create_sklearn_test_pipeline(estimators=[('SVC', SVC(probability=True))],
    #                                                                   reference=self.test_reference), keep_splits=True,
    #                                         strategy="random", project=project)
    #
    #
    #     execution = self.create_execution(experiment,pipeline=experiment.pipeline)
    #
    #     run = self.create_run(execution=execution)
    #     train_range = list(range(1, 1000 + 1))
    #     test_range = list(range(1000, 1100 + 1))
    #     split = self.create_split(run=run, num=0, train_idx=train_range, val_idx=None,
    #                               test_idx=test_range, keep_splits=True)
    #     assert (isinstance(split, Split))
    #     assert (split.test_idx == test_range)
    #     assert (split.train_idx == train_range)
    #     # self.app.computations.put(split)
    #     #
    #     # splits = self.app.computations.list()
    #     #
    #     # assert len(splits) == 1
    #     #
    #     # assert split.id == splits[0].id


if __name__ == '__main__':
    unittest.main()
