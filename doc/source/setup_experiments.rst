Setting up Experiments
======================

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
an array of values that are to be used for hyperparameter optimization

2. Through decorators

.. include:: ../../examples/07_seeding/07_seeding.py

Multi-pipline, multi-data Experiments
-------------------------------------

Currently, not suppoted