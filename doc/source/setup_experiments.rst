Setting up
============

Reference Object
-----------------
Git is an important aspect of PaDRe because we need to version the source code for tracking the lifetime of
experiments. Sometimes, the code might come from another python package too. For us to keep track of the provenance,
we need a reference that points us to the source and this is done via the reference object. A reference object can
either be a Python package or a Python file. If a Python file is given as a reference it should be part of a git
repository. The source code git should be created by the user and this could be in any directory of the user's system
environment. And if it is a Python package, we obtain the information about the Python package by inspecting the package
and storing the package identifier and the function that is called for execution. The reference object is explicitly
specified only when using pure code to create experiments and projects, while when using decorators it is automatically
picked up by the PaDRe framework to resolve the required names and file objects into references.


Creating a Project in PaDRe
----------------------------
A project contains one or more experiments that are semantically grouped together. It could be either different
experimental methods working towards an identical goal, or many different experiments that are parts of a larger goal.

Parameters of a project are
- Name: Name of the project.
- Description: A short description that provides information about the project.
- Reference: A reference object that specifies how the project was created and how it is handled.


Creating an experiment in PaDRe
---------------------------------

An experiment requires the following parameters to be initialized.

-Name: The name of the experiment. This should be unique
-Description: A short description of the intention of the experiment.
-Dataset: The dataset on which the experiment will work on
-Pipeline: A workflow consisting of one or more algorithms
-Project: The project to which this experiment belongs to. The name of the project can be specified and PaDRe
automatically searches and groups the experiment under the specified project.
-strategy: Splitting strategy for the dataset. The supported strategies are random, cv for cross validation,
explicit where the user can explicitly specify the indices for training, testing and validation, function where the
user passes a function that returns the indices or index where a list of indices are passed. If no option is given,
the random splitting method is chosen.
-preprocessing\_pipeline: Preprocessing workflow for the dataset in a case that an algorithm has to be applied to the
dataset as a whole for the experiment. This could be something such as computing the mean and standard deviation of a
dataset or creating an embedding which normally should be based on the whole dataset.
-reference: A reference object to the source code being executed

Single Pipeline Experiments
---------------------------

Single pipeline experiments can be created in different ways:

1. Through class instantiation

.. code-block:: python

    from pypadre.core.model.project import Project
    from pypadre.core.model.experiment import Experiment
    from pypadre.binding.metrics import sklearn_metrics
    print(sklearn_metrics)

    self.app.datasets.load_defaults()
    project = Project(name='Sample Project',
                      description=Example Project',
                      creator=Function(fn=self.test_full_stack))

    _id = '_iris_dataset'
    dataset = self.app.datasets.list({'name': _id})

    experiment = Experiment(name='Sample Experiment', description='Example Experiment',
                            dataset=dataset.pop(), project=project,
                            pipeline=SKLearnPipeline(pipeline_fn=create_test_pipeline_multiple_estimators),
                            reference=self.test_full_stack)
    experiment.execute()

2. Through decorators

.. code-block:: python

    import numpy as np
    from sklearn.datasets import load_iris

    # Import the metrics to register them
    from pypadre.binding.metrics import sklearn_metrics
    from pypadre.examples.base_example import example_app

    app = example_app()


    @app.dataset(name="iris",
                 columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                          'petal width (cm)', 'class'], target_features='class')
    def dataset():
        data = load_iris().data
        target = load_iris().target.reshape(-1, 1)
        return np.append(data, target, axis=1)


    @app.experiment(dataset=dataset, reference_git=__file__,
                    experiment_name="Iris SVC - User Defined Metrics", seed=1, allow_metrics=True, project_name="Examples")
    def experiment():
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC
        estimators = [('SVC', SVC(probability=True))]
        return Pipeline(estimators)


Hyperparameter Optimization
---------------------------
1. Through parameters passed to the experiment execute function. The parameters are passed as a dictionary with the
key as the component name and an inner dictionary. The inner dictionary contains the parameter name as the key and
an array of values that are to be used for hyperparameter optimization.

.. code-block:: python


    parameter_dict = {'SVR': {'C': [0.1, 0.2]}}
    experiment.execute(parameters={'SKLearnEvaluator': {'write_results': True}, 'SKLearnEstimator': {'parameters': parameter_dict}


2. Through decorators using the parameter keyword


.. code-block:: python

    @app.dataset(name="iris", columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                       'petal width (cm)', 'class'], target_features='class')
    def dataset():
        data = load_iris().data
        target = load_iris().target.reshape(-1, 1)
        return np.append(data, target, axis=1)

    @app.parameter_map()

    def parameters():
        return {'SKLearnEstimator': {'parameters': {'SVC': {'C': [0.1, 0.5, 1.0]}, 'PCA': {'n_components': [1, 2, 3]}}}}

    @app.experiment(dataset=dataset, reference_package=__file__, parameters=parameters, experiment_name="Iris SVC",
                    project_name="Examples", ptype=SKLearnPipeline)
    def experiment():
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC
        estimators = [('PCA', PCA()), ('SVC', SVC(probability=True))]
        return Pipeline(estimators)

Multi-pipline, multi-data Experiments
-------------------------------------

Currently, not supported